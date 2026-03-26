from abc import ABC, abstractmethod
from .vectorobservation import VectorObservation
from .vectoragent import VectorAgent
from .league import League

from typing import Optional, List, Tuple

import numpy as np
try:
    import gym
except ImportError:
    import gymnasium as gym
import torch
import copy


class PlayerException(Exception):
    """Raise when players in the environment are incorrectly set."""


class VectorMultiAgentEnv(ABC):
    """
    Base class for vectorized multi-agent environments.
    Supports league-based opponent selection for competitive self-play.
    """

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 ego_ind: int = 0,
                 n_players: int = 2,
                 partner: VectorAgent = None,
                 league: League = None):
        self.num_envs = num_envs
        self.device = device
        self.ego_ind = ego_ind
        self.n_players = n_players
        self.partner = partner
        self.league = league

        self._obs = tuple()
        self._actions = None

        # Random first-person-view assignment per environment
        self.env_fpv_player_ids = torch.randint(
            0, self.n_players, (self.num_envs,), device=self.device)

        self.env_opponent_ids = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int)
        self.env_game_played_vs_current_opponent = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int)

        self.merged_actions = torch.zeros(
            (self.n_players, self.num_envs, 1), device=self.device)
        assert self.n_players == 2

    def getDummyEnv(self, player_num: int):
        return self

    def add_partner_agent(self, agent: VectorAgent) -> None:
        self.partner = agent

    def _get_actions(self, obs, ego_act=None, validation=False):
        partner_obs = self.merge_observations(obs, fpv=False)
        partner = self.partner
        assert self.n_players == 2

        if hasattr(partner, "agent"):
            # League-based opponent: load weights per opponent ID
            partner_actions = torch.zeros((self.num_envs, 1)).to(self.device).long()
            for one_oppo_ids in torch.unique(self.env_opponent_ids[partner_obs.active.bool()]):
                oppo_id_int = int(one_oppo_ids.detach().cpu().numpy())
                if oppo_id_int == -1:
                    # Self-play: copy ego weights to partner
                    partner.agent.load_state_dict(self.ego.agent.state_dict())
                else:
                    one_opponent_weight = self.league.get_weight(oppo_id_int)
                    partner.agent.load_state_dict(one_opponent_weight)
                to_rollout_slots = torch.logical_and(
                    partner_obs.active,
                    (self.env_opponent_ids == one_oppo_ids).bool())
                partner_actions[to_rollout_slots, 0] = partner.get_action_partner(
                    partner_obs, to_rollout_slots)
        else:
            partner_actions = torch.zeros((self.num_envs, 1)).to(self.device).long()
            partner_actions[:, 0] = partner.get_action(partner_obs, record=False)

        stacked_action = torch.stack([ego_act, partner_actions])
        self.action_indices = torch.stack(
            [self.env_fpv_player_ids, 1 - self.env_fpv_player_ids], dim=1)
        self._actions = torch.unsqueeze(
            torch.gather(stacked_action[:, :, 0].T, 1, self.action_indices).T, 2)

        return self._actions

    def _update_players(self, rews, done):
        nextrew = self.merge_rewards(rews, fpv=False)
        self.partner.update(nextrew, done)

    def step(self, action: torch.Tensor, validation=False):
        acts = self._get_actions(self._obs, action, validation=validation)
        self._obs, rews, done, info = self.n_step(acts)

        self._update_players(rews, done)

        ego_obs = self.merge_observations(self._obs, fpv=True)
        ego_rew = self.merge_rewards(rews, fpv=True)
        ego_done = done

        # League win/loss tracking
        if self.league is not None:
            rwds_update = ego_rew[done.bool()]
            if len(rwds_update) > 0:
                oppid_update = self.env_opponent_ids[done.bool()]
                self.env_game_played_vs_current_opponent[done.bool()] += 1
                for one_rwd, one_oppid in zip(
                        rwds_update.detach().cpu().numpy(),
                        oppid_update.detach().cpu().numpy()):
                    if one_oppid == -1:
                        self.league.update_result(one_oppid, (one_rwd + 1) / 2, selfplay=True)
                    else:
                        self.league.update_result(one_oppid, (one_rwd + 1) / 2)

                # Resample opponents every 10 games
                if (self.env_game_played_vs_current_opponent >= 10).sum() > 0:
                    mask = self.env_game_played_vs_current_opponent >= 10
                    self.env_opponent_ids[mask] = torch.from_numpy(
                        self.league.select_opponent_batch(mask.sum().cpu().int())
                    ).int().to(self.env_opponent_ids.device)
                    self.env_game_played_vs_current_opponent[mask] = 0

        return ego_obs, ego_rew, ego_done, info

    def merge_rewards(self, rwd, fpv=True):
        if fpv:
            return torch.gather(
                rwd.T, 1,
                self.env_fpv_player_ids.reshape([self.num_envs, 1])
            ).reshape([self.num_envs])
        else:
            return torch.gather(
                rwd.T, 1,
                (1 - self.env_fpv_player_ids).reshape([self.num_envs, 1])
            ).reshape([self.num_envs])

    def merge_observations(self, obs, fpv=True):
        if fpv:
            self._merged_obs.active[self.env_fpv_player_ids == 0] = obs[0].active[self.env_fpv_player_ids == 0]
            self._merged_obs.active[self.env_fpv_player_ids == 1] = obs[1].active[self.env_fpv_player_ids == 1]
            self._merged_obs.obs[self.env_fpv_player_ids == 0] = obs[0].obs[self.env_fpv_player_ids == 0]
            self._merged_obs.obs[self.env_fpv_player_ids == 1] = obs[1].obs[self.env_fpv_player_ids == 1]
            self._merged_obs.state[self.env_fpv_player_ids == 0] = obs[0].state[self.env_fpv_player_ids == 0]
            self._merged_obs.state[self.env_fpv_player_ids == 1] = obs[1].state[self.env_fpv_player_ids == 1]
            self._merged_obs.action_mask[self.env_fpv_player_ids == 0] = obs[0].action_mask[self.env_fpv_player_ids == 0]
            self._merged_obs.action_mask[self.env_fpv_player_ids == 1] = obs[1].action_mask[self.env_fpv_player_ids == 1]
        else:
            self._merged_obs.active[self.env_fpv_player_ids == 0] = obs[1].active[self.env_fpv_player_ids == 0]
            self._merged_obs.active[self.env_fpv_player_ids == 1] = obs[0].active[self.env_fpv_player_ids == 1]
            self._merged_obs.obs[self.env_fpv_player_ids == 0] = obs[1].obs[self.env_fpv_player_ids == 0]
            self._merged_obs.obs[self.env_fpv_player_ids == 1] = obs[0].obs[self.env_fpv_player_ids == 1]
            self._merged_obs.state[self.env_fpv_player_ids == 0] = obs[1].state[self.env_fpv_player_ids == 0]
            self._merged_obs.state[self.env_fpv_player_ids == 1] = obs[0].state[self.env_fpv_player_ids == 1]
            self._merged_obs.action_mask[self.env_fpv_player_ids == 0] = obs[1].action_mask[self.env_fpv_player_ids == 0]
            self._merged_obs.action_mask[self.env_fpv_player_ids == 1] = obs[0].action_mask[self.env_fpv_player_ids == 1]
        return self._merged_obs

    def reset(self):
        self._obs = self.n_reset()
        self._merged_obs = copy.deepcopy(self._obs[0])
        ego_obs = self.merge_observations(self._obs, fpv=True)
        return ego_obs

    @abstractmethod
    def n_step(self, actions: torch.Tensor):
        """Perform actions and return (observations, rewards, done, info)."""

    @abstractmethod
    def n_reset(self):
        """Reset and return list of VectorObservations per player."""

    def close(self, **kwargs):
        pass


def to_torch(a):
    return a.detach().clone()


class MadronaEnv(VectorMultiAgentEnv):
    """Wraps a Madrona C++ simulator into a vectorized multi-agent environment."""

    def __init__(self, num_envs, gpu_id, sim, debug_compile=True,
                 obs_size=None, state_size=None, discrete_action_size=None,
                 env_device=None, league=None):
        self.sim = sim

        self.static_dones = self.sim.done_tensor().to_torch()
        self.static_active_agents = self.sim.active_agent_tensor().to_torch()
        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_agent_states = self.sim.agent_state_tensor().to_torch()
        self.static_action_masks = self.sim.action_mask_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()
        self.static_worldID = self.sim.world_id_tensor().to_torch().to(torch.long)
        self.static_agentID = self.sim.agent_id_tensor().to_torch().to(torch.long)

        self.obs_size = self.static_observations.shape[2] if obs_size is None else obs_size
        self.state_size = self.static_agent_states.shape[2] if state_size is None else state_size
        self.discrete_action_size = self.static_action_masks.shape[2] if discrete_action_size is None else discrete_action_size

        # Scattered tensors for proper agent<->world mapping
        self.static_scattered_active_agents = self.static_active_agents.detach().clone()
        self.static_scattered_observations = self.static_observations.detach().clone()
        self.static_scattered_agent_states = self.static_agent_states.detach().clone()
        self.static_scattered_action_masks = self.static_action_masks.detach().clone()
        self.static_scattered_rewards = self.static_rewards.detach().clone()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        if env_device is None:
            env_device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        super().__init__(num_envs, device=env_device,
                         n_players=self.static_observations.shape[0], league=league)

        self.infos = [{}] * self.num_envs

    def to_torch(self, a):
        return a.to(self.device)

    def n_step(self, actions):
        actions_device = self.static_agentID.get_device()
        actions = actions.to(actions_device if actions_device != -1 else torch.device('cpu'))
        self.static_actions.copy_(actions[self.static_agentID, self.static_worldID, :])

        self.sim.step()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        obs = [VectorObservation(
            self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
            self.to_torch(self.static_scattered_observations[i, :, :self.obs_size]),
            self.to_torch(self.static_scattered_agent_states[i, :, :self.state_size]),
            self.to_torch(self.static_scattered_action_masks[i, :, :self.discrete_action_size].to(torch.bool)))
               for i in range(self.n_players)]

        return obs, self.to_torch(self.static_scattered_rewards), self.to_torch(self.static_dones), self.infos

    def n_reset(self):
        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        obs = [VectorObservation(
            self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
            self.to_torch(self.static_scattered_observations[i, :, :self.obs_size]),
            self.to_torch(self.static_scattered_agent_states[i, :, :self.state_size]),
            self.to_torch(self.static_scattered_action_masks[i, :, :self.discrete_action_size].to(torch.bool)))
               for i in range(self.n_players)]
        return obs

    def close(self, **kwargs):
        pass
