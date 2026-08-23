# AWAC Friend Recommendation — Actor-Critic with Feature Weighting

A reinforcement-learning approach to link prediction / friend recommendation in social network graphs, using an Actor-Critic architecture with a learned, partially-weighted combination of structural node features (degree, closeness, betweenness, PageRank, neighborhood similarity) plus pairwise link-prediction features (Jaccard similarity, common neighbors, delta-modularity).

> **Development note:** This project was built in a *vibe-coding* style — fast, exploratory, AI-assisted iteration focused on getting an end-to-end RL link-prediction pipeline running, rather than a fully test-driven design from the start. See below for what was found and fixed during review, and what remains a known limitation.

## Problem

Given a social network graph, predict which currently-unconnected node pairs are most likely to become connected (i.e., recommend new friends), using a proper train/test **edge** split — unlike a simpler baseline that might just split users.

## Approach

- **State (9 features per candidate pair u→v)**: `degree_u, closeness_u, betweenness_u, pagerank_u, ns_u (neighborhood similarity), degree_v, jaccard(u,v), common_neighbors(u,v), delta_modularity(u,v)`.
- **Model**: an Actor-Critic network. A learned weighting layer scales the 5 structural features of the source node `u` (final weight = `1 + w_raw`, centered at 1); the 4 pairwise features pass through unweighted. L1 regularization is applied to 4 of the 5 raw weights; `ns_u`'s weight is instead regularized toward 1.
- **Training**: each episode samples a random node `u`, scores all valid candidate targets, samples an action from the resulting policy, and updates via an advantage-actor-critic loss (`policy_loss + value_loss + regularization`). The reward is `delta_modularity` of the chosen pair.
- **Evaluation**: proper **edge-level** train/test split (20% of edges held out). Held-out edges are positives; an equal number of random non-edges are negatives. Standard recommender metrics are reported **@k** for multiple k values: Precision, Recall, F1, Accuracy, Hit Rate, NDCG, Diversity, Coverage, AUC-ROC, AUC-PR.

## Tech stack

Python · PyTorch · NetworkX · scikit-learn · matplotlib

## Project structure

```
AWAC Friend recommendation/
├── src/
│   └── main.py              # features, Actor-Critic model, training loop, link-prediction evaluation, plots
├── data/
│   └── sample/                # small sample graph (edge list) — optional
├── outputs/                    # generated plots (git-ignored, created at runtime)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/main.py --graph data/network.txt --episodes 2000 --topk 5 10 20
```

Key CLI options:

| Flag | Default | Description |
|---|---|---|
| `--graph` | `data/network.txt` | Path to input edge-list graph |
| `--episodes` | 2000 | Training episodes per top-k scenario |
| `--topk` | 5 10 20 | Which top-k values to train/evaluate separately |
| `--eval-every` | 50 | Evaluate metrics every N episodes |
| `--lr` | 1e-3 | Learning rate |
| `--gamma` | 0.99 | Discount factor (not currently used in the single-step reward, kept for future multi-step extension) |
| `--l1-lambda` | 1e-4 | L1 regularization strength on 4 of the 5 structural feature weights |
| `--ns-reg-alpha` | 1e-3 | Regularization strength pulling `ns_u`'s weight toward 1 |
| `--output-dir` | `outputs` | Where plots are saved |

**Outputs** (saved as PNGs in `outputs/`, one set per top-k scenario, plus a combined comparison across all k): per-metric trend plots (precision/recall/F1/accuracy/hit-rate/NDCG/diversity/coverage/AUC-ROC/AUC-PR), reward/value/advantage trends, feature-weight evolution over training, and a feature-importance bar chart.

## Possible extensions

- Align the training reward with the evaluation objective (e.g., reward based on whether the sampled pair is a real held-out test edge, sampled appropriately to avoid leakage)
- Add experience replay / mini-batch updates instead of one-sample-per-episode updates
- Compare against simple baselines (Jaccard-only or common-neighbors-only ranking) to quantify what the learned feature weighting actually adds
