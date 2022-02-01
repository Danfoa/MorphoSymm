import sys
from prefixed import Float as FloatSI
from collections import namedtuple
from dataclasses import dataclass
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch.running_mean_std import RunningMeanStd

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os
import pathlib
from tqdm import tqdm

import logging
log = logging.getLogger(__name__)


@dataclass
class EpochTimeStats:
    """
    Class for storing stats of algorithm performance
    """
    env_step_time: float   # Time per single agent environment step (optimization/update + sim) [s/env_it]
    sim_step_time: float   # Time per single agent physics simulation step [s/sim_it]
    update_step_time: float  # Time per single agent physics simulation step [s/sim_it]
    epoch_time: float      # Epoch duration [s]
    epoch_steps: float     # Env steps in last epoch
    epoch_episodes: float  # Env episodes in last epoch

class SACEpochStats:

    actor_losses: list = []
    entropies: list = []
    alphas: list = []
    alpha_losses: list = []
    Q1_losses: list = []
    Q2_losses: list = []

    def append(self, actor_loss, entropy, alpha, alpha_loss, critic1_loss, critic2_loss):
        self.actor_losses.append(actor_loss)
        self.entropies.append(entropy)
        self.Q1_losses.append(critic1_loss)
        self.Q2_losses.append(critic2_loss)
        self.alphas.append(alpha)
        self.alpha_losses.append(alpha_loss)

    def as_dict(self):
        return {r'Loss(\pi(s))': self.actor_losses,
                r'Loss(Q_1(s,a))': self.Q1_losses,
                r'Loss(Q_2(s,a))': self.Q2_losses,
                r'\alpha': self.alphas,
                r'Loss(\alpha)': self.alpha_losses,
                r'Entropy': self.entropies}

    def __len__(self):
        return len(self.actor_losses)

class SACAgent:
    def __init__(self, base_name, config):
        print(config)

        # Get working directory and checkpoints dir path
        self.experiment_dir = config.get("experiment_dir", pathlib.Path(os.getcwd()))
        self.checkpoints_dir = pathlib.Path(self.experiment_dir, "nn")
        self.checkpoints_dir.mkdir(exist_ok=True)
        log.info(f"Checkpoints path {self.experiment_dir}")

        # TODO: Get obs shape and self.network
        self.base_init(base_name, config)
        self.num_seed_steps = config["num_seed_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = config["critic_tau"]
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = int(config["replay_buffer_size"])
        self.normalize_input = config.get("normalize_input", False)

        self.total_steps = config.get("total_steps", 1e7)  # temporary, in future we will use other approach
        self.max_episode_steps = config.get("max_episode_steps", 'inf')   # Experience samples in an episode
        self.max_episode_steps = np.inf if self.max_episode_steps == 'inf' else self.max_episode_steps
        self.log_frequency = config["log_frequency"]  # Experience samples in an epoch

        print(self.batch_size, self.num_envs, self.num_envs)

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self.sac_device)
        self.log_alpha.requires_grad = True
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num': self.actions_num,
            'input_shape': obs_shape
        }
        self.model = self.network.build(net_config)
        self.model.to(self.sac_device)

        print("Number of Agents", self.num_envs, "Batch Size", self.batch_size)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=self.config['actor_lr'],
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))

        self.critic_optimizer = torch.optim.Adam(self.model.sac_network.critic.parameters(),
                                                 lr=self.config["critic_lr"],
                                                 betas=self.config.get("critic_betas", [0.9, 0.999]))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.config["alpha_lr"],
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

        self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                               self.env_info['action_space'].shape,
                                                               self.replay_buffer_size,
                                                               self.sac_device)
        self.target_entropy_coef = config.get("target_entropy_coef", 1.0)
        self.target_entropy = self.target_entropy_coef * -self.env_info['action_space'].shape[0]
        print("Target entropy", self.target_entropy)
        self.cumulative_steps = 0   # Number of individual agent experience samples/steps performed
        self.cumulative_episodes = 0
        self.algo_observer = config['features']['observer']

        # TODO: Is there a better way to get the maximum number of episodes?
        # self.max_episodes = torch.ones(self.num_actors, device=self.sac_device)*self.num_steps_per_episode
        # self.episode_lengths = np.zeros(self.num_actors, dtype=int)
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.sac_device)

    def base_init(self, base_name, config):
        self.config = config
        self.env_config = config.get('env_config', {})
        self.num_envs = config.get('num_envs', 1)       # Number of envs running in parallel

        self.env_info = config.get('env_info')
        self.env_name = config['env_name']
        print("Env name:", self.env_name)
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_envs, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self.sac_device = config.get('device', 'cuda:0')
        # TODO: Remove PPO var name dependency:
        self.ppo_device = self.sac_device
        print(f'Env info:\n{self.env_info}')

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.weight_decay = config.get('weight_decay', 0.0)
        # self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.c_loss = nn.MSELoss()
        # self.c2_loss = nn.SmoothL1Loss()

        self.save_best_after_n_steps = config.get('save_best_after_n_steps', 1e5)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.obs_shape = self.observation_space.shape

        # Running avg metrics
        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.sac_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.sac_device)

        # self.min_alpha = torch.tensor(np.log(1)).float().to(self.sac_device)

        self.update_time = 0    # Iteration frequency for optimizing critic and actor
        self.last_mean_reward = -100500
        self.play_time = 0.
        self.epoch_num = 0.

        self.writer = SummaryWriter(self.experiment_dir.joinpath('tb_log' + datetime.now().strftime("_%d-%H_solo-%M-%S")))
        log.info(f"Tensorboard log dir {self.writer.log_dir}")

        self.is_tensor_obses = None
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            obs_torch_dtype = torch.uint8
        else:
            obs_torch_dtype = torch.float32

        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sac_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.sac_device)
        self.obs = torch.empty(size=self.observation_space.shape, dtype=obs_torch_dtype)
        self.dones = torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.sac_device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self.sac_device

    def get_full_state_weights(self):
        state = self.get_weights()

        state['steps'] = self.cumulative_steps
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()

        return state

    def get_weights(self):
        state = {'actor': self.model.sac_network.actor.state_dict(),
                 'critic': self.model.sac_network.critic.state_dict(),
                 'critic_target': self.model.sac_network.critic_target.state_dict()}
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input:
            self.running_mean_std.load_state_dict(weights['running_mean_std'])

    def set_full_state_weights(self, weights):
        self.set_weights(weights)

        self.cumulative_steps = weights['step']
        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + (not_done * self.gamma * (target_Q - self.alpha * log_prob))
            target_Q = target_Q.detach()

        # get current basis estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)

        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update_actor_and_temp(self, obs):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        dist = self.model.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True).mean()
        critic_Q1, critic_Q2 = self.model.critic(obs, action)
        critic_Q = torch.min(critic_Q1, critic_Q2)

        actor_loss = (self.alpha.detach() * log_prob - critic_Q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        if self.learnable_temperature:
            alpha_loss = (-self.alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        return actor_loss.detach(), entropy.detach(), self.alpha.detach(), alpha_loss  # TODO: maybe not self.alpha

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        not_done = ~done

        obs = self.preproc_obs(obs)
        next_obs = self.preproc_obs(next_obs)

        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, reward, next_obs, not_done)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_temp(obs)

        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                self.critic_tau)
        return actor_loss, entropy, alpha, alpha_loss, critic1_loss, critic2_loss

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        if self.normalize_input:
            obs = self.running_mean_std(obs)
        return obs

    def env_step(self, actions):
        obs, rewards, dones, infos = self.vec_env.step(actions)  # (obs_space) -> (n, obs_space)

        self.cumulative_steps += self.num_envs
        if self.is_tensor_obses:
            return obs, rewards, dones, infos
        else:
            return torch.from_numpy(obs).to(self.sac_device), torch.from_numpy(rewards).to(
                self.sac_device), torch.from_numpy(dones).to(self.sac_device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()
            if isinstance(obs, dict):
                obs = obs['obs']
        if self.is_tensor_obses is None:
            self.is_tensor_obses = torch.is_tensor(obs)
            print("Observations are tensors:", self.is_tensor_obses)

        if self.is_tensor_obses:
            return obs.to(self.sac_device)
        else:
            return torch.from_numpy(obs).to(self.sac_device)

    def act(self, obs, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if sample else dist.mean
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2
        return actions

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_losses is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def train_epoch(self) -> [EpochTimeStats, SACEpochStats]:
        """
        Collect agents experience on the environment and optimize the critic and actor functions for an entire epoch
        (i.e., user defined number of single agent simulation steps) or until max_sim_steps is reached.
        """
        random_exploration = self.cumulative_steps < self.num_seed_steps   # Act randomly first to fill replay buffer

        epoch_start_time = time.time()
        update_time, env_step_time = 0., 0.
        sac_epoch_stats = SACEpochStats()

        obs = self.obs

        prev_steps = self.cumulative_steps
        prev_episodes = self.cumulative_episodes
        sim_steps_per_epoch = int(self.log_frequency / self.num_envs)
        for _ in range(sim_steps_per_epoch):  # Epoch
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_envs, *self.env_info["action_space"].shape), device=self.sac_device) * 2 - 1
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), sample=True)

            step_start = time.time()
            with torch.no_grad():
                next_obs, rewards, dones, infos = self.env_step(action)
            self.pbar.update(self.num_envs)

            env_step_time += time.time() - step_start

            if isinstance(obs, dict):  # TODO: Why is dict returned?
                obs = obs['obs']
            if isinstance(next_obs, dict):
                next_obs = next_obs['obs']

            self.episode_rewards += rewards
            self.episode_lengths += 1

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_envs]
            self.cumulative_episodes += torch.sum(dones).item()  # Count the number of episodes.
            # For episodes that ended, save metric of final reward value
            self.game_rewards.update(self.episode_rewards[done_indices])
            self.game_lengths.update(self.episode_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            # If episodes have finite step lengths, terminate them.
            if np.isfinite(self.max_episode_steps):
                dones = dones * (self.episode_lengths != self.max_episode_steps)
                # TODO: Episodes terminated with max_steps should not be stored with basis or V values as rewards estimates.
            # TODO: Check for infinite obs values -> terminate those episodes
            # no_finite = torch.all(torch.isfinite(next_obs), dim=-1)

            # Reset counters of rewards and steps for envs with episodes that terminated
            self.episode_rewards = self.episode_rewards * not_dones
            self.episode_lengths = self.episode_lengths * not_dones

            rewards = self.rewards_shaper(rewards)

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(dones, 1))

            self.obs = obs = next_obs.clone()

            if not random_exploration:
                self.set_train()
                update_start_time = time.time()
                actor_loss, entropy, alpha, alpha_loss, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time += time.time() - update_start_time

                sac_epoch_stats.append(actor_loss, entropy, alpha, alpha_loss, critic1_loss, critic2_loss)
            else:
                update_time = 0

        epoch_time = time.time() - epoch_start_time
        epoch_steps = (self.cumulative_steps - prev_steps)
        epoch_episodes = (self.cumulative_episodes - prev_episodes)
        env_step_time = epoch_time / epoch_steps
        sim_step_time = (epoch_time - update_time) / epoch_steps
        update_step_time = update_time / epoch_steps

        algo_stats = EpochTimeStats(env_step_time, sim_step_time, update_step_time,
                                    epoch_time, epoch_steps, epoch_episodes)
        return algo_stats, sac_epoch_stats

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        self.last_mean_reward = -100500
        last_step_saved = 0
        total_time = 0
        # rep_count = 0
        self.cumulative_steps, self.cumulative_episodes = 0, 0
        self.obs = self.env_reset()

        # Visualization
        self.pbar = tqdm(total=self.total_steps, disable=not self.print_stats, dynamic_ncols=True, maxinterval=20,
                         file=sys.stdout, position=0, leave=True, desc=self.config['name'],
                         unit=" it", unit_scale=True)
        self.pbar.monitor_interval = 120
        self.pbar.colour = 'blue'

        while True:
            self.epoch_num += 1

            epoch_stats, sac_epoch_stats = self.train_epoch()

            total_time += epoch_stats.epoch_time

            mean_reward = self.game_rewards.get_mean().item()
            mean_ep_length = self.game_lengths.get_mean().item()
            self.pbar.set_postfix_str(f'Ep:{FloatSI(self.cumulative_episodes):.1h}-'
                                      f'step/Ep:{FloatSI(mean_ep_length):.1h}-'
                                      f'rew:{FloatSI(mean_reward):.1h}', refresh=False)

            # Log stats
            self.write_stats(epoch_stats, sac_epoch_stats, mean_reward, mean_ep_length,
                             step=self.cumulative_steps, time=total_time)

            if self.game_rewards.current_size > 0:
                if mean_reward > self.last_mean_reward and self.cumulative_steps - last_step_saved >= self.save_best_after_n_steps:
                    last_step_saved = self.cumulative_steps
                    log.info(f'saving next best rewards: {mean_reward:.2f}')
                    self.last_mean_reward = mean_reward
                    self.save(os.path.join(self.checkpoints_dir, self.config['name']))
                    if self.last_mean_reward > self.config.get('score_to_win', float('inf')):
                        print('Network won!')
                        self.save(os.path.join(self.checkpoints_dir,
                                               self.config['name'] + 'ep=' + str(self.epoch_num) + 'rew=' + str(
                                                   mean_reward)))
                        break
                if self.cumulative_steps >= self.total_steps:
                    ckpt_name = 'last_' + self.config['name'] + 'ep=' + str(self.epoch_num) + 'rew=' + str(mean_reward)
                    self.save(os.path.join(self.checkpoints_dir, ckpt_name))
                    log.info('Maximum number of experience samples reached.')
                    break

        self.pbar.close()
        return self.last_mean_reward, self.epoch_num

    def write_stats(self, epoch_stats: EpochTimeStats, sac_epoch_stats: SACEpochStats,
                    mean_reward: float, mean_ep_length: float, step, time):
        self.writer.add_scalar('performance/env_step_time/step', epoch_stats.env_step_time, step)
        self.writer.add_scalar('performance/sim_step_time/step', epoch_stats.sim_step_time, step)
        self.writer.add_scalar('performance/update_step_time/step', epoch_stats.update_step_time, step)

        if self.cumulative_steps > self.num_seed_steps and len(sac_epoch_stats) > 0:
            self.writer.add_scalar('losses/pi_loss/step', torch_ext.mean_list(sac_epoch_stats.actor_losses).item(), step)
            self.writer.add_scalar('losses/Q1_loss/step', torch_ext.mean_list(sac_epoch_stats.Q1_losses).item(), step)
            self.writer.add_scalar('losses/Q2_loss/step', torch_ext.mean_list(sac_epoch_stats.Q2_losses).item(), step)
            self.writer.add_scalar('losses/entropy/step', torch_ext.mean_list(sac_epoch_stats.entropies).item(), step)
            if not self.learnable_temperature:
                self.writer.add_scalar('losses/alpha_loss/step', torch_ext.mean_list(sac_epoch_stats.alpha_losses).item(), step)
            self.writer.add_scalar('info/alpha/step', torch_ext.mean_list(sac_epoch_stats.alphas).item(), step)

        self.writer.add_scalar('info/epochs/step', self.epoch_num, step)

        self.algo_observer.after_print_stats(step, self.epoch_num, time)

        if self.game_rewards.current_size > 0:
            self.writer.add_scalar('rewards/step', mean_reward, step)
            self.writer.add_scalar('rewards/time', mean_reward, time)
            self.writer.add_scalar('episode_lengths/step', mean_ep_length, step)
            self.writer.add_scalar('episode_lengths/time', mean_ep_length, time)
