"""
Gomoku Simple Self-Play Training
=================================

Trains a Gomoku agent using PPO with simple self-play (no league).
The ego agent always plays against a copy of itself (shared policy).

This is simpler than league training and good for initial experimentation.

Usage:
    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gomoku_cache python train_selfplay.py \
        --num-envs 256 --num-steps 128 --num-updates 2000 --cuda
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gomoku_env import GomokuMadrona
from pantheonrl_extension.vectoragent import CleanPPOAgent, RandomVectorAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Gomoku Self-Play Training")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num-res-blocks", type=int, default=4)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="gomoku_selfplay_output")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", default=False)
    return parser.parse_args()


class MirrorPartner:
    """
    A simple partner that mirrors the ego agent's policy.
    For self-play without league overhead.
    """
    def __init__(self, ego_agent):
        self.ego = ego_agent

    def get_action(self, obs, record=False):
        with torch.no_grad():
            action = torch.zeros((obs.obs.shape[0],)).to(obs.obs.device).long()
            if not obs.active.logical_not().all():
                action[obs.active], _, _, _ = self.ego.agent.get_action_and_value(
                    obs.obs[obs.active].float(),
                    obs.obs[obs.active].float(),
                    obs.action_mask[obs.active])
        return action

    def update(self, rewards, dones):
        pass


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    use_env_cpu = (device.type == "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    envs = GomokuMadrona(
        num_envs=args.num_envs,
        gpu_id=args.gpu_id,
        debug_compile=False,
        use_cpu=args.use_cpu,
        use_env_cpu=use_env_cpu,
    )

    ego = CleanPPOAgent(
        envs=envs,
        name="gomoku_selfplay",
        device=device,
        num_updates=args.num_updates,
        verbose=True,
        lr=args.learning_rate,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )

    partner = MirrorPartner(ego)
    envs.add_partner_agent(partner)

    obs = envs.reset()
    start_time = time.time()
    total_games = 0

    print(f"Starting Gomoku self-play training on {device}")
    print(f"  Environments: {args.num_envs}, Steps: {args.num_steps}")

    for update in range(1, args.num_updates + 1):
        for step in range(args.num_steps):
            action = ego.get_action(obs)
            obs, reward, done, info = envs.step(action)
            ego.update(reward, done)
            total_games += done.sum().item()

        if update % 10 == 0:
            elapsed = time.time() - start_time
            fps = (update * args.num_steps * args.num_envs) / elapsed
            print(f"Update {update}/{args.num_updates} | FPS: {fps:.0f} | Games: {total_games}")

        if update % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, "models", f"agent_{update}.pt")
            torch.save(ego.agent.state_dict(), save_path)
            print(f"  Saved: {save_path}")

    final_path = os.path.join(args.output_dir, "models", "agent_final.pt")
    torch.save(ego.agent.state_dict(), final_path)
    print(f"Training complete! Final model: {final_path}")


if __name__ == "__main__":
    main()
