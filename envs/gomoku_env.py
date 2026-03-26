try:
    from gym.spaces import Discrete, MultiBinary
except ImportError:
    from gymnasium.spaces import Discrete, MultiBinary

from pantheonrl_extension.vectorenv import MadronaEnv, VectorObservation, VectorAgent

import build.madrona_gomoku_python as gomoku_python

import numpy as np
import torch

DEFAULT_N = 2

DEFAULT_CONFIG = {
    "players": DEFAULT_N,
}

# Gomoku: 15x15 board
# Observation: 225 board cells + 1 current player = 226
# Action: 225 discrete (row * 15 + col)

OBS_SIZE = 226
ACTION_SIZE = 225


class GomokuMadrona(MadronaEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, config=None,
                 use_cpu=False, use_env_cpu=False, league=None):
        self.config = (config if config is not None else DEFAULT_CONFIG)

        sim = gomoku_python.GomokuSimulator(
            exec_mode=gomoku_python.madrona.ExecMode.CPU if use_cpu else gomoku_python.madrona.ExecMode.CUDA,
            gpu_id=gpu_id,
            num_worlds=num_envs,
            players=self.config["players"],
            debug_compile=debug_compile,
        )

        self.observation_space = MultiBinary(OBS_SIZE)
        self.action_space = Discrete(ACTION_SIZE)
        self.share_observation_space = MultiBinary(OBS_SIZE)

        device = None
        if use_env_cpu:
            device = torch.device('cpu')

        super().__init__(
            num_envs=num_envs,
            gpu_id=gpu_id,
            sim=sim,
            debug_compile=debug_compile,
            obs_size=OBS_SIZE,
            state_size=OBS_SIZE,
            discrete_action_size=ACTION_SIZE,
            env_device=device,
            league=league,
        )
