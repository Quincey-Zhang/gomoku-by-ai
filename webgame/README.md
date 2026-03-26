# Gomoku Web Game

Play Gomoku (Five in a Row) against a trained AI in your browser.

## Usage

```bash
conda activate madrona
cd Gomoku/webgame

# Auto-detect latest model from training output
python server.py

# Or specify a model path
python server.py --model-path ../train/gomoku_league_output/models/agent_final.pt

# Custom host/port
python server.py --host 0.0.0.0 --port 8080
```

Then open http://localhost:5000 in your browser.

## Features

- Play as Black (first) or White (second)
- Stochastic or deterministic AI mode
- Undo moves
- Win/loss/draw tracking
- AI value estimate display
- Highlighted last move and winning line
