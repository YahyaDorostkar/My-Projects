# RL-Based Friend Recommendation (Deep Q-Learning)

A reinforcement-learning approach to friend recommendation in social network graphs. A DQN agent learns to recommend new connections for a user by optimizing a reward signal built from network centrality measures (degree, closeness, betweenness) of the user's evolving "friend ring."

> **Development note:** This project was built in a *vibe-coding* style — fast, exploratory, AI-assisted iteration focused on getting an end-to-end RL pipeline running, rather than a fully test-driven design from the start. It's a solid proof-of-concept, not production-hardened code. Known limitations are listed below.

## Problem

Given a social network graph, recommend a new connection for a target user such that the recommendation improves the structural quality of their local network neighborhood (centrality, closeness, betweenness).

## Approach

- **Environment**: each user's current "friend ring" (neighbors) defines the state. An action is recommending one other user as a new connection.
- **Reward**: computed from how centrality/closeness/betweenness/modularity change for the user's friend ring after adding the recommended connection.
- **Agent**: a small feed-forward DQN (2 hidden layers) trained with an epsilon-greedy policy; multiple epsilon (exploration rate) values are compared side by side.
- **Evaluation**: for held-out test users, the agent's top recommendation is checked against the user's real graph neighbors to compute accuracy/precision/recall/F1.

## Tech stack

Python · PyTorch · NetworkX · scikit-learn · matplotlib

## Project structure

```
RL-based Friend recommendation/
├── src/
│   └── main.py              # environment, DQN model, training loop, evaluation, plots
├── data/
│   └── sample/               # small sample graph (e.g. dolphins.gml) — optional
├── outputs/                   # generated plots (git-ignored, created at runtime)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/main.py --graph data/yelp.txt --episodes 2000
```

Key CLI options (all optional, sensible defaults included):

| Flag | Default | Description |
|---|---|---|
| `--graph` | `data/yelp.txt` | Path to input `.gml` graph |
| `--episodes` | 2000 | Training episodes per epsilon scenario |
| `--interval` | 100 | Evaluate metrics every N episodes |
| `--gamma` | 0.9 | Discount factor |
| `--epsilon-values` | 0.01 0.05 0.1 0.2 0.5 | Exploration rates compared |
| `--epsilon-decay` | 0.995 | Epsilon decay per episode |
| `--lr` | 0.001 | Learning rate |
| `--output-dir` | `outputs` | Where plots are saved |

**Outputs** (saved as PNGs in `outputs/`): accuracy trend plots, per-epsilon reward trends, and Q-value trends across episodes.

## Fixes applied to the original vibe-coded version

These bugs were present in the first draft and have since been fixed (verified with a test run — see below):

- **Reward function's "mixed result" case returned the same +1 as a full improvement.** Now returns `0.5`, so partial and full improvements are distinguishable in the training signal.
- **Tensor shape mismatch warning during loss computation** (target vs. prediction shapes didn't match, causing incorrect broadcasting). Fixed by aligning both to scalar tensors.
- **Hardcoded Google Colab path** replaced with a `--graph` CLI argument, and the loader now supports both `.gml` files and plain edge-list `.txt` files (e.g. `dolphins.txt`).

**Verified working**: the full pipeline (`src/main.py`) was run end-to-end on a small test graph with no errors and no warnings, producing all 6 output plots correctly.

## Still-known limitations

- **Single-step episodes**: each episode effectively evaluates one recommendation rather than a multi-step interaction sequence (loop caps at `step_count < 10` but `step()` always returns `done=True`).
- The modularity proxy is a structural heuristic, not the formal mathematical per-node contribution to Newman's modularity Q.

## Possible extensions

- Extend episodes to multi-step recommendation sequences instead of single-shot
- Add a baseline (e.g. common-neighbors or Adamic-Adar link prediction) to compare against the RL agent
- Track train/test performance over time to check for overfitting on small graphs


