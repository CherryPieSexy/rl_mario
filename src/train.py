import argparse
from functools import partial

import torch
import torch.multiprocessing as mp

from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.recurrent_encoders import OneLayerActorCritic
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.ppo import PPO

import cherry_rl.algorithms.parallel as parallel

from src.encoder import Encoder
from src.environment import make_mario_env

recurrent = False

emb_size = 512
action_size = 7
distribution_str = 'Categorical'

gamma = 0.9
train_env_num = 32
rollout_len = 64

ac_args = {'input_size': emb_size, 'action_size': action_size}
ppo_args = {
    'normalize_adv': True,
    'returns_estimator': 'v-trace',
    'ppo_n_epoch': 8, 'ppo_n_mini_batches': 4,
    'rollback_alpha': 0.0
}
train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 250, 'rollout_len': rollout_len}

run_test_process = True
render_test_env = True
test_process_act_deterministic = False


def make_env(world, level):
    def make():
        return make_mario_env(world, level)
    return make


def make_ac_model(ac_device):
    def make_ac():
        return OneLayerActorCritic(**ac_args)
    model = AgentModel(
        ac_device, make_ac, distribution_str,
        make_obs_encoder=Encoder,
        value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', '-w', type=int, default=1)
    parser.add_argument('--level', '-l', type=int, default=1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--entropy', '-e', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = f'logs/mario/world-{args.world}-level-{args.level}_tune_1/'
    create_log_dir(log_dir, __file__)
    device = torch.device(args.device)

    ppo_args.update({'entropy': args.entropy, 'learning_rate': args.learning_rate})
    training_args.update({'log_dir': log_dir})

    parallel.run(
        log_dir, partial(make_env, world=args.world, level=args.level),
        make_ac_model, device,
        make_optimizer, train_args, training_args,
        run_test_process=run_test_process,
        render_test_env=render_test_env,
        test_process_act_deterministic=test_process_act_deterministic
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
