# Gomoku (Five in a Row) - Madrona RL Environment

A GPU-accelerated Gomoku (Five in a Row) reinforcement learning environment built on the [Madrona game engine](https://madrona-engine.github.io/), with league-based self-play training.

## Game Rules

- **Board**: 15x15 grid
- **Players**: 2 (Black and White, alternating turns)
- **Objective**: Place 5 stones in a row (horizontal, vertical, or diagonal)
- **Draw**: Board fills up with no winner

## Architecture

### C++ Simulation Core (`src/gomoku_env/`)

The game logic runs inside Madrona's ECS framework, enabling thousands of parallel games on GPU:

| File | Description |
|------|-------------|
| `sim.hpp/cpp` | Core game logic: action placement, win detection, observation encoding |
| `mgr.hpp/cpp` | Manager bridging C++ and Python, CPU and GPU execution backends |
| `bindings.cpp` | Python bindings via nanobind (`GomokuSimulator`) |
| `init.hpp` | World initialization structures |
| `rng.hpp` | Random number generator |

**Key design**: Observations are ego-centric (own stones = 1, opponent = 2), so a single shared policy works for both players.

### Neural Network (`pantheonrl_extension/vectoragent.py`)

**GomokuResNetCNN** — AlphaZero-inspired architecture:
- Input: 226-dim obs → one-hot encode to 3-channel 15x15 image
- Trunk: 3x3 conv (→128ch) + 6 residual blocks (128ch each, BatchNorm + ReLU)
- Policy head: 1x1 conv (→32ch) → FC → 225 actions
- Value head: 1x1 conv (→1ch) → FC(256) → FC(1)

### League Training (`train/train_league.py`)

Uses **Prioritized Fictitious Self-Play (PFSP)**:
1. Train ego agent via PPO against opponents sampled from a league pool
2. League opponent selection is weighted toward policies the ego struggles against
3. When ego's win rate exceeds threshold (default 55%) against all opponents, archive current weights
4. Mix of ~30% self-play + ~70% league opponents

### Python Environment Wrapper (`envs/gomoku_env.py`)

Wraps the C++ simulator with gym-compatible spaces:
- Observation space: `MultiBinary(226)` — 225 board cells + 1 player indicator
- Action space: `Discrete(225)` — position = row * 15 + col
- Action masking: occupied cells are masked out

## Requirements

- CUDA >= 12.0 (for GPU mode)
- CMake >= 3.18
- Python >= 3.10
- Conda (miniconda/anaconda)

## Installation

```bash
conda create -n gomoku python=3.10
conda activate gomoku
pip install torch numpy tensorboard gym

git clone <this-repo> Gomoku
cd Gomoku
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j
cd ..

pip install -e .
```

> On some systems, specify the CUDA toolkit:
> ```bash
> cmake -D CUDAToolkit_ROOT=/usr/local/cuda-12.0 ..
> ```

## Training

### League Training (recommended)

```bash
cd train

# GPU training
MADRONA_MWGPU_KERNEL_CACHE=/tmp/gomoku_cache python train_league.py \
    --num-envs 1000 --num-steps 128 --num-updates 5000 \
    --learning-rate 2.5e-4 --cuda \
    --selfplay-ratio 0.3 --archive-threshold 0.55

# CPU training (slower, for testing)
python train_league.py --num-envs 32 --num-steps 64 --num-updates 500
```

### Simple Self-Play

```bash
python train_selfplay.py --num-envs 256 --num-steps 128 --num-updates 2000 --cuda
```

## Evaluation

```bash
# Evaluate against random opponent
python evaluate.py --model-path gomoku_league_output/models/agent_final.pt \
    --num-games 1000 --opponent random --cuda

# Self-play evaluation
python evaluate.py --model-path gomoku_league_output/models/agent_final.pt \
    --num-games 1000 --opponent self --cuda

# Display games in text mode
python evaluate.py --model-path gomoku_league_output/models/agent_final.pt \
    --num-envs 1 --opponent random --display
```

## Project Structure

```
Gomoku/
├── CMakeLists.txt              # Top-level build
├── setup.py                    # Python package setup
├── __init__.py                 # Package init
├── README.md
├── external/
│   └── madrona/                # Madrona engine (git submodule)
├── src/
│   └── gomoku_env/             # C++ simulation core
│       ├── sim.hpp/cpp         # Game logic (ECS systems)
│       ├── mgr.hpp/cpp         # CPU/GPU manager
│       ├── bindings.cpp        # Python bindings
│       ├── init.hpp            # World init structs
│       ├── rng.hpp             # RNG utility
│       └── CMakeLists.txt      # Build config
├── envs/
│   └── gomoku_env.py           # Python env wrapper
├── pantheonrl_extension/
│   ├── vectorenv.py            # Vectorized multi-agent env base
│   ├── vectoragent.py          # PPO agent + GomokuResNetCNN
│   ├── vectorobservation.py    # Observation dataclass
│   └── league.py               # League (PFSP) manager
└── train/
    ├── train_league.py         # League training script
    ├── train_selfplay.py       # Simple self-play training
    └── evaluate.py             # Evaluation script
```
