"""
Gomoku League Training Script
==============================

Trains a Gomoku agent using PPO with league-based self-play.

The league maintains a pool of historical policy snapshots. During training:
1. The ego agent plays against opponents sampled from the league (PFSP)
2. A fraction of games are pure self-play (ego vs current ego weights)
3. When the ego agent's win rate exceeds a threshold against all opponents,
   the current weights are archived into the league
4. Training continues with the expanded opponent pool

This produces increasingly strong agents that are robust against diverse
play styles rather than overfitting to a single opponent.

Usage:
    # GPU training with league
    MADRONA_MWGPU_KERNEL_CACHE=/tmp/gomoku_cache python train_league.py \
        --num-envs 1000 --num-steps 128 --num-updates 5000 \
        --learning-rate 2.5e-4 --cuda

    # CPU training (slower, for testing)
    python train_league.py --num-envs 32 --num-steps 64 --num-updates 500
"""

import argparse
import os
import sys
import time
import copy
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gomoku_env import GomokuMadrona
from pantheonrl_extension.league import League
from pantheonrl_extension.vectoragent import CleanPPOAgent, GomokuResNetCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Gomoku League Training")

    # Environment
    parser.add_argument("--num-envs", type=int, default=2048,
                        help="Number of parallel environments")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--use-cpu", action="store_true", default=False,
                        help="Use CPU execution for Madrona simulator")

    # PPO hyperparameters
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of steps per rollout")
    parser.add_argument("--num-updates", type=int, default=int(1e6),
                        help="Total number of PPO updates")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--anneal-lr", action="store_true", default=True,
                        help="Anneal learning rate over training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.999,
                        help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of minibatches for PPO")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Number of PPO epochs per update")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=1e-2,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm for clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="Target KL divergence for early stopping")

    # League parameters
    parser.add_argument("--selfplay-ratio", type=float, default=0.2,
                        help="Fraction of games that are self-play")
    parser.add_argument("--archive-threshold", type=float, default=0.9,
                        help="Win rate threshold to archive a new policy")
    parser.add_argument("--archive-interval", type=int, default=10,
                        help="Minimum updates between archiving policies")
    parser.add_argument("--winrate-window", type=int, default=500,
                        help="EMA window size for win rate tracking")

    # Network architecture
    parser.add_argument("--num-res-blocks", type=int, default=4,
                        help="Number of residual blocks in the CNN")
    parser.add_argument("--num-channels", type=int, default=64,
                        help="Number of channels in the CNN")

    # Output
    parser.add_argument("--output-dir", type=str, default="gomoku_league_output",
                        help="Directory for saving models and logs")
    parser.add_argument("--save-interval", type=int, default=int(1e4),
                        help="Save model every N updates")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA for training")

    return parser.parse_args()


def get_agent_weights(agent):
    """Extract a copy of the agent's network weights as OrderedDict."""
    return OrderedDict(
        {k: v.clone().cpu() for k, v in agent.agent.state_dict().items()}
    )


def main():
    args = parse_args()

    # Setup
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    use_env_cpu = (device.type == "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(args.output_dir, "runs"))

    # Save args
    with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
        f.write(str(vars(args)))

    # Initialize league
    league = League(
        n=args.winrate_window,
        last_num=1000,
        output_dir=args.output_dir,
        selfplay_ratio=args.selfplay_ratio,
    )

    # Create environment
    envs = GomokuMadrona(
        num_envs=args.num_envs,
        gpu_id=args.gpu_id,
        debug_compile=False,
        use_cpu=args.use_cpu,
        use_env_cpu=use_env_cpu,
        league=league,
    )

    # Create ego agent (the one being trained)
    ego = CleanPPOAgent(
        envs=envs,
        name="gomoku_ego",
        device=device,
        num_updates=args.num_updates,
        verbose=True,
        lr=args.learning_rate,
        num_steps=args.num_steps,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )

    # Create partner agent (opponent, weights loaded from league)
    partner = CleanPPOAgent(
        envs=envs,
        name="gomoku_partner",
        device=device,
        num_updates=args.num_updates,
        verbose=False,
        lr=args.learning_rate,
        num_steps=args.num_steps,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )

    # Store reference to ego on env for self-play weight copying
    envs.ego = ego
    envs.add_partner_agent(partner)

    # Initialize league with initial (random) weights
    initial_weights = get_agent_weights(ego)
    league.add_weight(initial_weights, dump=True)

    # Initialize all environments to self-play
    envs.env_opponent_ids[:] = -1

    # Training loop
    print(f"Starting Gomoku league training on {device}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Steps/rollout: {args.num_steps}")
    print(f"  Total updates: {args.num_updates}")
    print(f"  ResNet blocks: {args.num_res_blocks}, channels: {args.num_channels}")
    print(f"  League selfplay ratio: {args.selfplay_ratio}")
    print(f"  Archive threshold: {args.archive_threshold}")
    print()

    obs = envs.reset()
    start_time = time.time()
    last_archive_update = 0
    total_games = 0

    for update in range(1, args.num_updates + 1):
        # Collect rollout
        for step in range(args.num_steps):
            action = ego.get_action(obs)
            obs, reward, done, info = envs.step(action)
            ego.update(reward, done)

            total_games += done.sum().item()

        # Log progress
        elapsed = time.time() - start_time
        fps = (update * args.num_steps * args.num_envs) / elapsed

        if update % 10 == 0:
            # Get league stats
            if league.initized() and hasattr(league, 'selfplay_winrate'):
                sp_wr = league.selfplay_winrate.v
                league_size = league.weight_number()
                avg_wr = np.mean([wr.v for wr in league.winrates]) if league.winrates else 0.5

                print(f"Update {update}/{args.num_updates} | "
                      f"FPS: {fps:.0f} | "
                      f"Games: {total_games} | "
                      f"League size: {league_size} | "
                      f"Self-play WR: {sp_wr:.3f} | "
                      f"Avg WR vs league: {avg_wr:.3f}")

                writer.add_scalar("league/selfplay_winrate", sp_wr, update)
                writer.add_scalar("league/avg_winrate", avg_wr, update)
                writer.add_scalar("league/size", league_size, update)
                writer.add_scalar("charts/total_games", total_games, update)
                writer.add_scalar("charts/FPS", fps, update)

        # Archive policy if win rate is high enough
        can_archive = update - last_archive_update >= args.archive_interval
        wr_match = league.initized() and league.winrate_all_match(args.archive_threshold)
        if can_archive and wr_match:
            new_weights = get_agent_weights(ego)
            league.add_weight(new_weights, dump=True)
            last_archive_update = update
            print(f"  >> Archived policy #{league.current_pid} at update {update}")
        elif update % 10 == 0 and league.initized():
            wr_vals = [(i.v, i.n) for i in league.winrates[-10:]]
            sp_wr = league.selfplay_winrate
            print(f"  [archive check] can_archive={can_archive}, wr_match={wr_match}")
            print(f"    self-play: WR={sp_wr.v:.3f} games={sp_wr.n}")
            for pid, (wr, n) in zip(league.pids[-10:], wr_vals):
                marker = ">" if wr > args.archive_threshold else " "
                print(f"   {marker} c_{pid}: WR={wr:.3f} games={n}")

            # Save league stats
            league.get_statics_table(dump=True)

        # Save model periodically
        if update % args.save_interval == 0:
            league.save_weight(get_agent_weights(ego))
            print(f"  Saved interval checkpoint at update {update}")

    # Final save
    league.save_weight(get_agent_weights(ego))
    league.get_statics_table(dump=True)

    print(f"\nTraining complete!")
    print(f"  Total updates: {args.num_updates}")
    print(f"  Total games: {total_games}")
    print(f"  League size: {league.weight_number()}")

    writer.close()


if __name__ == "__main__":
    main()
