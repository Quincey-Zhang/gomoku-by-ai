"""
Gomoku Web Game Server
======================

A Flask server that serves a web-based Gomoku game where a human
plays against a trained AI model.

Usage:
    python server.py --model-path ../train/gomoku_league_output/models/interval_1.pkl
    python server.py --model-dir ../train/gomoku_league_output/models/  # auto-pick latest

Then open http://localhost:5000 in your browser.
"""

import argparse
import os
import sys
import glob
import json
import pickle

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pantheonrl_extension.vectoragent import GomokuResNetCNN

app = Flask(__name__, static_folder='.', static_url_path='')

# Global model reference
agent = None
device = None

BOARD_H = 15
BOARD_W = 15
NUM_ACTIONS = BOARD_H * BOARD_W


def find_latest_model(model_dir):
    """Find the latest interval_N.pkl, falling back to league_N.pkl."""
    for prefix in ("interval_", "league_"):
        files = glob.glob(os.path.join(model_dir, f"{prefix}*.pkl"))
        if files:
            files.sort(key=lambda f: int(os.path.basename(f).replace(prefix, "").replace(".pkl", "")))
            return files[-1]
    return None


class DummyEnvSpace:
    """Mimics env spaces for network construction."""
    def __init__(self):
        self.n = NUM_ACTIONS
        self.shape = (NUM_ACTIONS,)


class DummyEnv:
    def __init__(self):
        self.action_space = DummyEnvSpace()
        self.observation_space = type('Space', (), {'shape': (NUM_ACTIONS + 1,)})()
        self.share_observation_space = type('Space', (), {'shape': (NUM_ACTIONS + 1,)})()


def load_model(model_path):
    """Load the trained GomokuResNetCNN model."""
    global agent, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_env = DummyEnv()
    agent = GomokuResNetCNN(dummy_env).to(device)
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"Loaded model from {model_path} on {device}")


def board_to_observation(board, current_player):
    """
    Convert board state to ego-centric observation tensor.
    board: list of 225 ints (0=empty, 1=black, 2=white)
    current_player: 1=black, 2=white

    Returns: tensor of shape (1, 226)
    Encoding: 0=empty, 1=ego stone, 2=opponent stone
    """
    obs = []
    for cell in board:
        if cell == 0:
            obs.append(0)
        elif cell == current_player:
            obs.append(1)  # ego
        else:
            obs.append(2)  # opponent
    # Append current player indicator (0 or 1)
    obs.append(0 if current_player == 1 else 1)
    return torch.tensor([obs], dtype=torch.float32, device=device)


def get_action_mask(board):
    """Return action mask: 1 for empty cells, 0 for occupied."""
    mask = []
    for cell in board:
        mask.append(1 if cell == 0 else 0)
    return torch.tensor([mask], dtype=torch.bool, device=device)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/ai_move', methods=['POST'])
def ai_move():
    """
    Receive the board state, return the AI's move.

    Request JSON:
        board: list of 225 ints (0=empty, 1=human, 2=AI)
        ai_player: int (1 or 2) - which player the AI is

    Response JSON:
        action: int (0-224, row*15+col)
        row: int
        col: int
    """
    data = request.get_json()
    board = data['board']
    ai_player = data['ai_player']

    obs = board_to_observation(board, ai_player)
    mask = get_action_mask(board)

    with torch.no_grad():
        action, _, _, value = agent.get_action_and_value(
            obs, obs, mask)

    action_id = action.item()
    row = action_id // BOARD_W
    col = action_id % BOARD_W

    return jsonify({
        'action': action_id,
        'row': row,
        'col': col,
        'value': value.item(),
    })


@app.route('/ai_move_deterministic', methods=['POST'])
def ai_move_deterministic():
    """
    Same as ai_move but picks the highest-probability action (greedy).
    """
    data = request.get_json()
    board = data['board']
    ai_player = data['ai_player']

    obs = board_to_observation(board, ai_player)
    mask = get_action_mask(board)

    with torch.no_grad():
        trunk = agent._forward_trunk(obs)
        p = F.relu(agent.policy_bn(agent.policy_conv(trunk)))
        p = p.reshape(p.size(0), -1)
        logits = agent.policy_fc(p)
        logits[~mask] = -float('inf')
        action_id = logits.argmax(dim=1).item()

        v = F.relu(agent.value_bn(agent.value_conv(trunk)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(agent.value_fc1(v))
        value = agent.value_fc2(v)

    row = action_id // BOARD_W
    col = action_id % BOARD_W

    return jsonify({
        'action': action_id,
        'row': row,
        'col': col,
        'value': value.item(),
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Gomoku Web Game Server")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a specific .pt model file")
    parser.add_argument("--model-dir", type=str,
                        default="../train/gomoku_league_output/models/",
                        help="Directory to search for the latest interval_N.pkl")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=7788,
                        help="Server port")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_model(args.model_dir)

    if model_path is None or not os.path.exists(model_path):
        print(f"Error: No model found. Provide --model-path or check --model-dir")
        print(f"  Searched: {args.model_dir}")
        sys.exit(1)

    load_model(model_path)
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
