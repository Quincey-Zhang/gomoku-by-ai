"""
Gomoku Evaluation Script
========================

Load a trained model and evaluate it against a random agent or itself.
Can also display games in text mode.

Usage:
    python evaluate.py --model-path output/models/agent_final.pt \
        --num-envs 100 --num-games 1000 --opponent random
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gomoku_env import GomokuMadrona
from pantheonrl_extension.vectoragent import GomokuResNetCNN
from pantheonrl_extension.vectorobservation import VectorObservation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gomoku Agent")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--opponent", type=str, default="self",
                        choices=["self", "random"],
                        help="Opponent type: 'self' or 'random'")
    parser.add_argument("--display", action="store_true", default=False,
                        help="Display games in text mode (only with --num-envs 1)")
    return parser.parse_args()


def display_board(obs_tensor):
    """Print the board state from an observation tensor."""
    board = obs_tensor[:225].reshape(15, 15).cpu().numpy()
    symbols = {0: '.', 1: 'X', 2: 'O'}
    print("   " + " ".join(f"{i:2d}" for i in range(15)))
    for r in range(15):
        row_str = " ".join(f" {symbols[int(board[r, c])]}" for c in range(15))
        print(f"{r:2d} {row_str}")
    print()


class SelfPlayPartner:
    def __init__(self, agent, device):
        self.agent = agent
        self.device = device

    def get_action(self, obs, record=False):
        with torch.no_grad():
            action = torch.zeros((obs.obs.shape[0],)).to(self.device).long()
            if not obs.active.logical_not().all():
                action[obs.active], _, _, _ = self.agent.get_action_and_value(
                    obs.obs[obs.active].float(),
                    obs.obs[obs.active].float(),
                    obs.action_mask[obs.active])
        return action

    def update(self, rewards, dones):
        pass


class RandomPartner:
    def __init__(self, device, num_actions=225):
        self.device = device
        self.num_actions = num_actions

    def get_action(self, obs, record=False):
        batch = obs.obs.shape[0]
        actions = torch.zeros(batch, dtype=torch.long, device=self.device)
        for i in range(batch):
            if obs.active[i]:
                valid = obs.action_mask[i].nonzero(as_tuple=True)[0]
                if len(valid) > 0:
                    actions[i] = valid[torch.randint(len(valid), (1,))]
        return actions

    def update(self, rewards, dones):
        pass


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    use_env_cpu = (device.type == "cpu")

    envs = GomokuMadrona(
        num_envs=args.num_envs,
        gpu_id=args.gpu_id,
        debug_compile=False,
        use_cpu=args.use_cpu,
        use_env_cpu=use_env_cpu,
    )

    # Load model
    agent = GomokuResNetCNN(envs).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"Loaded model from {args.model_path}")

    # Setup opponent
    if args.opponent == "self":
        partner = SelfPlayPartner(agent, device)
        print("Opponent: self-play")
    else:
        partner = RandomPartner(device)
        print("Opponent: random")

    envs.add_partner_agent(partner)

    # Evaluate
    obs = envs.reset()
    wins = 0
    losses = 0
    draws = 0
    total = 0

    print(f"Evaluating over {args.num_games} games...")

    while total < args.num_games:
        with torch.no_grad():
            action = torch.zeros((args.num_envs, 1), dtype=torch.long, device=device)
            if not obs.active.logical_not().all():
                act, _, _, _ = agent.get_action_and_value(
                    obs.obs[obs.active].float(),
                    obs.obs[obs.active].float(),
                    obs.action_mask[obs.active])
                action[obs.active, 0] = act

        obs, reward, done, info = envs.step(action)
        partner.update(reward, done)

        if args.display and args.num_envs == 1 and obs.active[0]:
            display_board(obs.obs[0])

        if done.any():
            for i in range(args.num_envs):
                if done[i] and total < args.num_games:
                    r = reward[i].item()
                    if r > 0:
                        wins += 1
                    elif r < 0:
                        losses += 1
                    else:
                        draws += 1
                    total += 1

    print(f"\nResults ({total} games):")
    print(f"  Wins:   {wins} ({100*wins/total:.1f}%)")
    print(f"  Losses: {losses} ({100*losses/total:.1f}%)")
    print(f"  Draws:  {draws} ({100*draws/total:.1f}%)")
    print(f"  Win rate: {(wins + 0.5*draws)/total:.3f}")


if __name__ == "__main__":
    main()
