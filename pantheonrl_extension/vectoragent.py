from abc import ABC, abstractmethod
from .vectorobservation import VectorObservation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class VectorAgent(ABC):
    @abstractmethod
    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        """
        Return an action given an observation.
        """

    @abstractmethod
    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Add new rewards and done information if the agent can learn.
        """


class RandomVectorAgent(VectorAgent):
    def __init__(self, sampler):
        self.sampler = sampler

    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        return self.sampler()

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        return


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class GomokuResNetCNN(nn.Module):
    """
    ResNet-based CNN for 15x15 Gomoku.

    Input: 226-dim vector -> reshape to 3-channel 15x15 image via one-hot encoding
      Channel 0: empty cells
      Channel 1: ego stones
      Channel 2: opponent stones

    Architecture:
      - Initial conv: 3 -> 128 channels, 3x3, padding 1
      - 6 residual blocks (128 channels each)
      - Policy head: 1x1 conv -> FC -> 225 actions
      - Value head: 1x1 conv -> FC -> FC -> 1 value

    This is inspired by AlphaZero's architecture but scaled down
    for efficient PPO training.
    """
    def __init__(self, envs, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.board_h = 15
        self.board_w = 15
        self.num_actions = envs.action_space.n

        # Initial convolution
        self.conv_init = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = layer_init(nn.Linear(32 * self.board_h * self.board_w, self.num_actions), std=0.01)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = layer_init(nn.Linear(self.board_h * self.board_w, 256))
        self.value_fc2 = layer_init(nn.Linear(256, 1), std=1.0)

    def _encode_board(self, x):
        """Convert flat observation to 3-channel board image."""
        # x: (batch, 226) — last element is current_player, first 225 are board
        obs = x[:, :-1]  # (batch, 225)
        board = obs.reshape(-1, self.board_h, self.board_w).to(torch.int64)
        # One-hot: 0->empty, 1->ego, 2->opponent → 3 channels
        board_onehot = F.one_hot(board, num_classes=3).float()
        # (batch, H, W, 3) -> (batch, 3, H, W)
        board_onehot = board_onehot.permute(0, 3, 1, 2)
        return board_onehot

    def _forward_trunk(self, x):
        """Shared trunk: initial conv + residual blocks."""
        board = self._encode_board(x)
        out = F.relu(self.bn_init(self.conv_init(board)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def get_value(self, x):
        trunk = self._forward_trunk(x)
        v = F.relu(self.value_bn(self.value_conv(trunk)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        return v

    def get_action_and_value(self, x, state, action_mask, action=None):
        trunk = self._forward_trunk(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(trunk)))
        p = p.reshape(p.size(0), -1)
        logits = self.policy_fc(p)

        # Mask invalid actions
        logits[torch.logical_not(action_mask)] = -float('inf')
        probs = Categorical(logits=logits)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(trunk)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        vals = self.value_fc2(v)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), vals


class CleanPPOAgent(VectorAgent):
    """
    PPO agent for Gomoku using the GomokuResNetCNN network.
    Supports league-based training with opponent weight loading.
    """
    def __init__(self,
                 envs,
                 name: str,
                 device: torch.device,
                 num_updates: int,
                 verbose: bool = True,
                 lr: float = 2.5e-4,
                 num_steps: int = 128,
                 anneal_lr: bool = True,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.2,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None,
                 num_res_blocks: int = 4,
                 num_channels: int = 64):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.name = name
        self.device = device
        self.verbose = verbose

        self.lr = lr
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.current_ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        if self.verbose and SummaryWriter is not None:
            self.writer = SummaryWriter(f"runs/{name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join(
                    [f"|{key}|{value}|" for key, value in vars(self).items()])),
            )
        else:
            self.writer = None

        self.agent = GomokuResNetCNN(self.envs, num_res_blocks=num_res_blocks, num_channels=num_channels).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        self.obs = torch.zeros((self.num_steps, self.num_envs) + envs.observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + envs.action_space.shape).to(device)
        self.action_masks = torch.zeros((self.num_steps, self.num_envs, envs.action_space.n)).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.active = torch.zeros((self.num_steps, self.num_envs), dtype=torch.bool).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)

        self.global_step = 0
        self.step = 0
        self.start_time = time.time()
        self.num_updates = num_updates
        self.updates = 1

        self.next_done = torch.zeros(self.num_envs, dtype=torch.bool).to(device)
        self.new_game = torch.zeros(self.num_envs, dtype=torch.bool).to(device)

        self.running_rewards = torch.zeros(self.num_envs).to(device)
        self.last_active = torch.zeros(self.num_envs, dtype=torch.long).to(device)

        self.mean_return_sum = 0
        self.num_returns = 0

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        dones = dones.to(dtype=torch.bool)
        self.running_rewards += rewards
        self.rewards[self.last_active] += torch.where(self.new_game, 0, rewards.view(-1))
        self.next_done |= dones

        if torch.any(dones):
            if self.verbose and self.writer:
                self.writer.add_scalar("charts/min_episodic_return", torch.min(self.running_rewards[dones]), self.global_step)
                self.writer.add_scalar("charts/max_episodic_return", torch.max(self.running_rewards[dones]), self.global_step)
            self.mean_return_sum += torch.mean(self.running_rewards[dones])
            self.num_returns += 1
            self.running_rewards[dones] = 0.0
            self.new_game[dones] = True

        self.step += 1
        self.global_step += 1

    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        if self.global_step > 0 and self.global_step % self.num_steps == 0 and record:
            self._ppo_update(obs)

        with torch.no_grad():
            action = torch.zeros((self.num_envs)).to(self.device).long()
            logprob = torch.zeros((self.num_envs)).to(self.device)
            value = torch.zeros((self.num_envs, 1)).to(self.device)
            if not obs.active.logical_not().all():
                action[obs.active], logprob[obs.active], _, value[obs.active] = \
                    self.agent.get_action_and_value(
                        obs.obs[obs.active].float(),
                        obs.obs[obs.active].float(),
                        obs.action_mask[obs.active])

            if record:
                self.values[self.step] = value.flatten()

        if record:
            self.obs[self.step] = obs.obs
            self.dones[self.step] = self.next_done
            self.active[self.step] = obs.active
            self.actions[self.step] = action
            self.action_masks[self.step] = obs.action_mask
            self.logprobs[self.step] = logprob

            self.next_done[:] = False
            self.rewards[self.step] = 0

            self.last_active[obs.active] = self.step
            self.new_game[obs.active] = False
        return action[:, None]

    def get_action_partner(self, obs: VectorObservation, rollout_slots) -> torch.tensor:
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                obs.obs[rollout_slots].float(),
                obs.obs[rollout_slots].float(),
                obs.action_mask[rollout_slots])
        return action

    def _ppo_update(self, obs):
        """Perform PPO update at the end of a rollout."""
        self.step = 0

        if self.anneal_lr:
            frac = 1.0 - (self.updates - 1.0) / self.num_updates
            lrnow = frac * self.lr
            self.current_ent_coef = frac * self.ent_coef
            self.optimizer.param_groups[0]["lr"] = lrnow

        with torch.no_grad():
            next_value = self.agent.get_value(obs.obs.float()).reshape(-1)
            advantages = torch.zeros_like(self.rewards).to(self.device)

            bootstrapped = obs.active.detach().clone().to(dtype=torch.bool)
            nextnonterminal = torch.zeros(self.num_envs).to(self.device)
            nextvalues = torch.zeros(self.num_envs).to(self.device)

            lastgaelam = torch.zeros(self.num_envs).to(self.device)
            nextnonterminal[bootstrapped] = 1.0 - self.next_done[bootstrapped].to(torch.float)
            nextvalues[bootstrapped] = next_value[bootstrapped]

            delta = torch.zeros(self.num_envs).to(self.device)

            for t in reversed(range(self.num_steps)):
                mask = self.active[t]
                computemask = mask & bootstrapped
                bootstrapped |= mask
                self.active[t] = computemask

                delta[computemask] = (self.rewards[t, computemask]
                                      + self.gamma * nextvalues[computemask] * nextnonterminal[computemask]
                                      - self.values[t, computemask])
                advantages[t, computemask] = lastgaelam[computemask] = (
                    delta[computemask]
                    + self.gamma * self.gae_lambda * nextnonterminal[computemask] * lastgaelam[computemask])

                nextnonterminal[mask] = 1.0 - self.dones[t, mask]
                nextvalues[mask] = self.values[t, mask]
            returns = advantages + self.values

        # Flatten
        b_obs = self.obs[self.active].reshape((-1,) + self.envs.observation_space.shape)
        b_logprobs = self.logprobs[self.active].reshape(-1)
        b_actions = self.actions[self.active].reshape((-1,) + self.envs.action_space.shape)
        b_advantages = advantages[self.active].reshape(-1)
        b_returns = returns[self.active].reshape(-1)
        b_values = self.values[self.active].reshape(-1)
        b_action_masks = self.action_masks[self.active].reshape((-1, self.envs.action_space.n))

        batch_size = b_obs.shape[0]
        if batch_size < 2:
            self.updates += 1
            return
        minibatch_size = max(batch_size // self.num_minibatches, 2)

        clipfracs = []
        for epoch in range(self.update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            epoch_kl = 0.0
            num_mb = 0
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_obs[mb_inds], b_action_masks[mb_inds], b_actions[mb_inds].long())
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    epoch_kl += approx_kl.item()
                    num_mb += 1
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv and mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.reshape(-1)
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.current_ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if num_mb > 0 and (epoch_kl / num_mb) > self.target_kl:
                    break

        if self.verbose:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.num_returns != 0:
                self.writer.add_scalar("charts/episodic_return", self.mean_return_sum / self.num_returns, self.global_step)
                self.mean_return_sum = 0
                self.num_returns = 0

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

        self.updates += 1
