"""
Gomoku AI vs AI Evaluation
===========================

Runs games between two trained AI models using a pure-Python Gomoku
engine (no Madrona simulator required).

Usage:
    # Same model vs itself
    python ai_vs_ai.py --model-a ../train/gomoku_league_output/models/league_288.pkl

    # Two different models
    python ai_vs_ai.py \
        --model-a ../train/gomoku_league_output/models/league_288.pkl \
        --model-b ../train/gomoku_league_output/models/league_100.pkl \
        --num-games 200

    # Show board after each game
    python ai_vs_ai.py \
        --model-a ../train/gomoku_league_output/models/league_288.pkl \
        --num-games 10 --display
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# ---------------------------------------------------------------------------
# Network definition (inlined from pantheonrl_extension/vectoragent.py
# to avoid the TensorBoard import chain)
# ---------------------------------------------------------------------------

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class GomokuResNetCNN(nn.Module):
    def __init__(self, envs, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.board_h   = 15
        self.board_w   = 15
        self.num_actions = envs.action_space.n

        self.conv_init = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init   = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([_ResBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = _layer_init(
            nn.Linear(32 * self.board_h * self.board_w, self.num_actions), std=0.01)

        self.value_conv  = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = _layer_init(nn.Linear(self.board_h * self.board_w, 256))
        self.value_fc2   = _layer_init(nn.Linear(256, 1), std=1.0)

    def _encode_board(self, x):
        obs   = x[:, :-1]
        board = obs.reshape(-1, self.board_h, self.board_w).to(torch.int64)
        board_onehot = F.one_hot(board, num_classes=3).float()
        return board_onehot.permute(0, 3, 1, 2)

    def _forward_trunk(self, x):
        out = F.relu(self.bn_init(self.conv_init(self._encode_board(x))))
        for block in self.res_blocks:
            out = block(out)
        return out

    def get_action_and_value(self, x, state, action_mask, action=None):
        trunk  = self._forward_trunk(x)
        p      = F.relu(self.policy_bn(self.policy_conv(trunk)))
        logits = self.policy_fc(p.reshape(p.size(0), -1))
        logits[torch.logical_not(action_mask)] = -float('inf')
        probs  = Categorical(logits=logits)

        v = F.relu(self.value_bn(self.value_conv(trunk)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        vals = self.value_fc2(v)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), vals

BOARD_H = 15
BOARD_W = 15
NUM_ACTIONS = BOARD_H * BOARD_W  # 225
OBS_SIZE = NUM_ACTIONS + 1       # 226


# ---------------------------------------------------------------------------
# Pure-Python Gomoku engine
# ---------------------------------------------------------------------------

class GomokuGame:
    """Single Gomoku game: 15x15 board, first to 5-in-a-row wins."""

    EMPTY = 0
    BLACK = 1   # player 0
    WHITE = 2   # player 1

    def __init__(self):
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
        self.current_player = 0   # 0=Black, 1=White
        self.move_count = 0
        self.done = False
        self.winner = None        # None=draw, 0=Black, 1=White

    def action_mask(self):
        """Return bool array of shape (225,): True for empty cells."""
        return (self.board.reshape(-1) == self.EMPTY)

    def observation(self, player):
        """
        226-dim ego-centric observation for `player`:
          cells 0..224 : 0=empty, 1=own stone, 2=opponent stone
          cell 225     : player id (0 or 1)
        """
        obs = np.zeros(OBS_SIZE, dtype=np.int8)
        flat = self.board.reshape(-1)
        own_stone   = self.BLACK if player == 0 else self.WHITE
        oppo_stone  = self.WHITE if player == 0 else self.BLACK
        obs[:NUM_ACTIONS][flat == own_stone]  = 1
        obs[:NUM_ACTIONS][flat == oppo_stone] = 2
        obs[NUM_ACTIONS] = player
        return obs

    def step(self, action):
        """
        Place stone at `action` (0..224).
        Returns: winner (0 or 1) or None if game continues / draw.
        Raises ValueError if action is invalid.
        """
        if self.done:
            raise RuntimeError("Game is already over.")

        row, col = divmod(action, BOARD_W)
        if self.board[row, col] != self.EMPTY:
            raise ValueError(f"Cell ({row},{col}) is already occupied.")

        stone = self.BLACK if self.current_player == 0 else self.WHITE
        self.board[row, col] = stone
        self.move_count += 1

        if self._check_win(row, col, stone):
            self.done = True
            self.winner = self.current_player
        elif self.move_count == NUM_ACTIONS:
            self.done = True   # draw, winner stays None

        self.current_player ^= 1
        return self.winner

    def _check_win(self, row, col, stone):
        """Check if placing `stone` at (row, col) creates 5-in-a-row."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < BOARD_H and 0 <= c < BOARD_W and self.board[r, c] == stone:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 5:
                return True
        return False

    def display(self, model_a_name="Model A", model_b_name="Model B"):
        """Print the board to stdout."""
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        header = f"  {'Black':^7} = X ({model_a_name if self.current_player == 1 else model_b_name})" \
                 f"   {'White':^7} = O ({model_b_name if self.current_player == 1 else model_a_name})"
        print(header)
        col_header = "    " + "".join(f"{c:<2}" for c in range(BOARD_W))
        print(col_header)
        for r in range(BOARD_H):
            row_str = "".join(f"{symbols[self.board[r, c]]} " for c in range(BOARD_W))
            print(f"{r:2d}  {row_str}")
        print()


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

class DummyEnvSpace:
    n = NUM_ACTIONS
    shape = (NUM_ACTIONS,)


class DummyEnv:
    action_space = DummyEnvSpace()
    observation_space = type('S', (), {'shape': (OBS_SIZE,)})()
    share_observation_space = type('S', (), {'shape': (OBS_SIZE,)})()


def load_model(path, device, num_res_blocks=4, num_channels=64):
    """Load a GomokuResNetCNN from a .pkl state-dict file."""
    model = GomokuResNetCNN(DummyEnv(), num_res_blocks=num_res_blocks,
                            num_channels=num_channels).to(device)
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def get_action(model, obs_np, mask_np, device, deterministic=False):
    """
    Run inference for a single position.
    obs_np  : (226,) int8 numpy array
    mask_np : (225,) bool numpy array
    Returns: action int (0..224)
    """
    obs  = torch.tensor(obs_np,  dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(mask_np, dtype=torch.bool,    device=device).unsqueeze(0)

    if deterministic:
        trunk  = model._forward_trunk(obs)
        p      = F.relu(model.policy_bn(model.policy_conv(trunk)))
        logits = model.policy_fc(p.reshape(p.size(0), -1))
        logits[~mask] = -float('inf')
        return logits.argmax(dim=1).item()
    else:
        action, _, _, _ = model.get_action_and_value(obs, obs, mask)
        return action.item()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def play_game(model_a, model_b, device, deterministic=False, display=False,
              model_a_name="Model A", model_b_name="Model B", first_player=0):
    """
    Play one game. `first_player` (0 or 1) decides which model plays Black.
      Black (player 0) => model_a if first_player==0, else model_b
      White (player 1) => model_b if first_player==0, else model_a

    Returns: 'a', 'b', or 'draw'
    """
    models = [model_a, model_b] if first_player == 0 else [model_b, model_a]
    game   = GomokuGame()

    while not game.done:
        player = game.current_player
        obs    = game.observation(player)
        mask   = game.action_mask()

        # Safety: if mask is all False (shouldn't happen), pick any cell
        if not mask.any():
            action = 0
        else:
            action = get_action(models[player], obs, mask, device, deterministic)
            # Fallback for illegal move (shouldn't happen with correct mask)
            if not mask[action]:
                action = int(np.where(mask)[0][0])

        game.step(action)

    if display:
        game.display(model_a_name, model_b_name)

    winner = game.winner  # 0=Black, 1=White, None=draw
    if winner is None:
        return 'draw'
    # Map back from board player to model
    winner_model = 'a' if (winner == first_player) else 'b'
    return winner_model


def evaluate(model_a, model_b, device, num_games, deterministic=False,
             display=False, model_a_name="Model A", model_b_name="Model B"):
    results = {'a': 0, 'b': 0, 'draw': 0}
    # Alternate who plays Black (first-mover advantage balancing)
    for i in range(num_games):
        first = i % 2  # 0 -> model_a is Black; 1 -> model_b is Black
        outcome = play_game(
            model_a, model_b, device,
            deterministic=deterministic,
            display=display,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            first_player=first,
        )
        results[outcome] += 1

        if (i + 1) % max(1, num_games // 10) == 0:
            done = i + 1
            a, b, d = results['a'], results['b'], results['draw']
            print(f"  [{done:>{len(str(num_games))}}/{num_games}]  "
                  f"{model_a_name} wins: {a}  {model_b_name} wins: {b}  Draws: {d}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Gomoku AI vs AI Evaluation")

    parser.add_argument("--model-a", type=str, required=True,
                        help="Path to model A .pkl file (required)")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Path to model B .pkl file (defaults to model A for self-play)")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to play (default: 100)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use greedy (argmax) action selection instead of sampling")
    parser.add_argument("--display", action="store_true", default=False,
                        help="Print the final board of every game")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA for inference")
    parser.add_argument("--num-res-blocks", type=int, default=4,
                        help="Residual blocks in the network (must match training, default: 4)")
    parser.add_argument("--num-channels", type=int, default=64,
                        help="Channels in the network (must match training, default: 64)")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    model_b_path = args.model_b if args.model_b else args.model_a
    self_play    = (args.model_b is None or args.model_b == args.model_a)

    model_a_name = os.path.splitext(os.path.basename(args.model_a))[0]
    model_b_name = ("self" if self_play
                    else os.path.splitext(os.path.basename(model_b_path))[0])

    print(f"Loading Model A: {args.model_a}")
    model_a = load_model(args.model_a, device,
                         num_res_blocks=args.num_res_blocks,
                         num_channels=args.num_channels)

    if self_play:
        print(f"Model B: same as A (self-play)")
        model_b = model_a
    else:
        print(f"Loading Model B: {model_b_path}")
        model_b = load_model(model_b_path, device,
                             num_res_blocks=args.num_res_blocks,
                             num_channels=args.num_channels)

    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"\nRunning {args.num_games} games ({mode}, device={device})")
    print(f"  Black/White alternate every game for fairness")
    print(f"  {model_a_name}  vs  {model_b_name}\n")

    results = evaluate(
        model_a, model_b, device,
        num_games=args.num_games,
        deterministic=args.deterministic,
        display=args.display,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
    )

    total = args.num_games
    a, b, d = results['a'], results['b'], results['draw']
    win_rate_a = (a + 0.5 * d) / total

    print(f"\n{'='*50}")
    print(f"Results  ({total} games)")
    print(f"{'='*50}")
    print(f"  {model_a_name:<30}  {a:>4} wins  ({100*a/total:5.1f}%)")
    print(f"  {model_b_name:<30}  {b:>4} wins  ({100*b/total:5.1f}%)")
    print(f"  Draws                          {d:>4}        ({100*d/total:5.1f}%)")
    print(f"{'='*50}")
    print(f"  {model_a_name} win rate (with draws split): {win_rate_a:.3f}")


if __name__ == "__main__":
    main()
