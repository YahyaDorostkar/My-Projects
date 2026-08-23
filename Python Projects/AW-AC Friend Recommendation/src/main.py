# -*- coding: utf-8 -*-
"""
RL-Based Friend Recommendation — Actor-Critic with Feature Weighting
=====================================================================
Trains an Actor-Critic agent for link prediction / friend recommendation.
The model learns per-feature weights over 5 structural features of the
source node (degree, closeness, betweenness, pagerank, neighborhood
similarity) combined with 4 unweighted pairwise features (target degree,
Jaccard similarity, common neighbors, delta-modularity).

Usage
-----
    python src/main.py --graph data/network.txt --episodes 2000 --topk 5 10 20

Note
----
This project was originally developed in a "vibe coding" style (rapid,
exploratory, AI-assisted iteration) rather than a fully test-driven
design. See the README for what was fixed and what remains a known
limitation before using it as a reference implementation.
"""

import argparse
import os
import random
import math
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


############################################
# Part 1: Graph & feature helper functions
############################################

def load_graph_from_file(path):
    """Expects a file with lines 'u v' (unweighted edges)."""
    G = nx.Graph()
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            G.add_edge(u, v)
    return G


def jaccard_similarity(G, u, v):
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    if not nu and not nv:
        return 0.0
    inter = len(nu & nv)
    union = len(nu | nv)
    return inter / union if union > 0 else 0.0


def num_common_neighbors(G, u, v):
    if not G.has_node(u) or not G.has_node(v):
        return 0
    return len(set(G.neighbors(u)) & set(G.neighbors(v)))


def neighborhood_similarity(G, u):
    """Average Jaccard(u, each neighbor of u). 0 if u has no neighbors."""
    if not G.has_node(u):
        return 0.0
    neigh = list(G.neighbors(u))
    if len(neigh) == 0:
        return 0.0
    sims = [jaccard_similarity(G, u, v) for v in neigh]
    return float(np.mean(sims)) if sims else 0.0


def delta_modularity(G, u, v):
    """Simplified proxy for the modularity change from adding edge (u,v):
    (deg_u + deg_v) / (2*n). Not the formal Newman modularity delta —
    documented as a simplification, consistent with the rest of the
    reward-shaping features in this project."""
    if not G.has_node(u) or not G.has_node(v):
        return 0.0
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return (deg_u + deg_v) / (2 * n)


############################################
# Part 2: Local/global feature cache
############################################

class LocalFeatureCache:
    """Caches per-node features: degree, closeness, betweenness, pagerank,
    neighborhood similarity (ns)."""

    def __init__(self, G: nx.Graph):
        self.G = G
        self.cache = {}
        self.global_centrality_computed = False

    def _compute_global_centralities(self):
        if self.G.number_of_nodes() == 0:
            self.global_centrality_computed = True
            return

        closeness_dict = nx.closeness_centrality(self.G)
        betweenness_dict = nx.betweenness_centrality(self.G, normalized=True)
        pagerank_dict = nx.pagerank(self.G)

        for n in self.G.nodes():
            if n not in self.cache:
                self.cache[n] = {}
            self.cache[n]['degree'] = float(self.G.degree(n))
            self.cache[n]['ns'] = float(neighborhood_similarity(self.G, n))
            self.cache[n]['closeness'] = float(closeness_dict.get(n, 0.0))
            self.cache[n]['betweenness'] = float(betweenness_dict.get(n, 0.0))
            self.cache[n]['pagerank'] = float(pagerank_dict.get(n, 0.0))

        self.global_centrality_computed = True

    def compute_features_for_nodes(self, nodes):
        nodes = list(set(nodes))
        if not self.global_centrality_computed:
            self._compute_global_centralities()

        for n in nodes:
            if n not in self.cache:
                self.cache[n] = {
                    'degree': float(self.G.degree(n)),
                    'ns': float(neighborhood_similarity(self.G, n)),
                    'closeness': 0.0,
                    'betweenness': 0.0,
                    'pagerank': 0.0,
                }

    def get_features(self, node):
        return self.cache.get(node, None)


############################################
# Part 3: 9-feature state vector
############################################

def build_state_vector(G, u, v, feature_cache: LocalFeatureCache):
    """
    state(u,v) = [degree_u, closeness_u, betweenness_u, pagerank_u, ns_u,
                  degree_v, jaccard(u,v), common_neighbors(u,v), delta_modularity(u,v)]
    """
    feature_cache.compute_features_for_nodes([u, v])
    fu = feature_cache.get_features(u) or {}
    fv = feature_cache.get_features(v) or {}

    fu.setdefault('degree', float(G.degree(u)))
    fv.setdefault('degree', float(G.degree(v)))
    fu.setdefault('closeness', 0.0)
    fu.setdefault('betweenness', 0.0)
    fu.setdefault('pagerank', 0.0)
    fu.setdefault('ns', float(neighborhood_similarity(G, u)))

    jacc = jaccard_similarity(G, u, v)
    cn = num_common_neighbors(G, u, v)
    dq = delta_modularity(G, u, v)

    state = np.array([
        fu['degree'], fu['closeness'], fu['betweenness'], fu['pagerank'], fu['ns'],
        fv['degree'], jacc, cn, dq,
    ], dtype=np.float32)

    return state, dq


############################################
# Part 4: Actor-Critic network with partial feature weighting
############################################

class ActorCriticNet(nn.Module):
    """
    A learned weighting layer is applied only to the first 5 features
    (u's structural features): [degree_u, closeness_u, betweenness_u,
    pagerank_u, ns_u]. The remaining 4 features
    [degree_v, jaccard, common_neighbors, delta_modularity] pass through
    unweighted.

    - Raw weight params (w_raw) are centered near 0; final weight = 1 + w_raw.
    - L1 regularization applies to the first 4 raw weights; ns_u is exempt
      (regularized toward w=1 instead, via a separate quadratic term).
    """

    def __init__(self, state_dim=9, num_weighted_features=5, hidden_dim=64, action_dim=1):
        super().__init__()
        self.state_dim = state_dim
        self.num_weighted_features = num_weighted_features

        self.feature_weights_raw = nn.Parameter(torch.zeros(num_weighted_features))
        self.feature_bias = nn.Parameter(torch.zeros(num_weighted_features))

        nn.init.normal_(self.feature_weights_raw, mean=0.0, std=0.1)
        with torch.no_grad():
            self.feature_weights_raw[4] = 0.0  # ns_u starts at weight exactly 1

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_weighted = x[:, :self.num_weighted_features]
        x_unweighted = x[:, self.num_weighted_features:]

        w = 1.0 + self.feature_weights_raw
        z_weighted = x_weighted * w + self.feature_bias
        z = torch.cat([z_weighted, x_unweighted], dim=1)

        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.actor_head(z), self.critic_head(z)


############################################
# Part 5: Evaluation metrics
############################################

def hit_rate_at_k(recommended, ground_truth_set, k):
    if len(recommended) == 0:
        return 0.0
    topk = recommended[:k]
    return 1.0 if any(item in ground_truth_set for item in topk) else 0.0


def ndcg_at_k(recommended, ground_truth_set, k):
    """Binary relevance NDCG@k."""
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in ground_truth_set)
    ideal_hits = min(len(ground_truth_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def diversity_at_k(G, recommended, k):
    """Average shortest-path distance between recommended items."""
    if len(recommended) < 2:
        return 0.0
    sub = recommended[:k]
    distances = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            u, v = sub[i], sub[j]
            if nx.has_path(G, u, v):
                try:
                    d = nx.shortest_path_length(G, u, v)
                except nx.NetworkXNoPath:
                    d = 0
            else:
                d = 0
            distances.append(d)
    return float(np.mean(distances)) if distances else 0.0


def coverage_at_k(all_items, recommended_lists, k):
    """Fraction of all_items that appear at least once across all top-k lists."""
    rec_items = set()
    for rec in recommended_lists:
        rec_items.update(rec[:k])
    return len(rec_items) / len(all_items) if all_items else 0.0


############################################
# Part 6: Train/test edge split
############################################

def train_test_edge_split(G, test_ratio=0.2, seed=42):
    random.seed(seed)
    edges = list(G.edges())
    random.shuffle(edges)
    split_idx = int(len(edges) * (1 - test_ratio))
    train_edges = edges[:split_idx]
    test_edges = edges[split_idx:]

    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    G_test = G.copy()
    return G_train, G_test, train_edges, test_edges


############################################
# Part 7: Actor-Critic training loop
############################################

def train_actor_critic(
    G,
    num_episodes=500,
    learning_rate=1e-3,
    gamma=0.99,
    topk_list=(5,),
    eval_every=50,
    device=None,
    l1_lambda=1e-4,
    ns_reg_alpha=1e-3,
):
    """Runs one training scenario for a given top-k evaluation setting."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_train, G_test, train_edges, test_edges = train_test_edge_split(G, test_ratio=0.2)
    nodes = list(G_train.nodes())

    model = ActorCriticNet(state_dim=9, num_weighted_features=5, hidden_dim=64, action_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    feature_cache = LocalFeatureCache(G_train)

    metrics_history = defaultdict(list)
    episodes_recorded = []
    episode_rewards, episode_values, episode_advantages = [], [], []
    feature_weights_history = []

    for episode in range(1, num_episodes + 1):
        model.train()
        u = random.choice(nodes)
        neighbors_u = set(G_train.neighbors(u))
        candidates = [v for v in nodes if v != u and v not in neighbors_u]
        if not candidates:
            continue

        states, dq_list = [], []
        for v in candidates:
            state_vec, dq = build_state_vector(G_train, u, v, feature_cache)
            states.append(state_vec)
            dq_list.append(dq)

        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).to(device)
        logits, values = model(states_tensor)
        logits = logits.view(-1)
        values = values.view(-1)

        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()

        reward = dq_list[action_idx]
        value_chosen = values[action_idx]
        advantage = reward - value_chosen.detach().item()

        log_prob = torch.log(probs[action_idx] + 1e-8)
        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(value_chosen, torch.tensor(reward, dtype=torch.float32, device=device))
        base_loss = policy_loss + value_loss

        l1_penalty = torch.norm(model.feature_weights_raw[:4], p=1)
        w_ns = 1.0 + model.feature_weights_raw[4]
        ns_reg = ns_reg_alpha * (w_ns - 1.0) ** 2

        loss = base_loss + l1_lambda * l1_penalty + ns_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(reward)
        episode_values.append(value_chosen.detach().item())
        episode_advantages.append(advantage)

        if episode % eval_every == 0:
            print(f"Episode {episode} / {num_episodes} ... evaluating...")
            eps_metrics = evaluate_link_prediction(G_train, G_test, model, feature_cache, topk_list=topk_list, device=device)
            episodes_recorded.append(episode)
            for key, val in eps_metrics.items():
                metrics_history[key].append(val)
            with torch.no_grad():
                fw = (1.0 + model.feature_weights_raw).detach().cpu().numpy().copy()
            feature_weights_history.append(fw)

    return model, dict(metrics_history), feature_weights_history, episodes_recorded, episode_rewards, episode_values, episode_advantages


############################################
# Part 8: Link-prediction evaluation
############################################

def evaluate_link_prediction(G_train, G_test, model, feature_cache: LocalFeatureCache, topk_list=(5,), device=None):
    """For each real test edge not present in G_train, treat as a positive
    sample; negatives are sampled from non-edges. The model scores state(u,v)
    and top-k recommendations are formed per source node."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_nodes = list(G_train.nodes())
    train_edges_set = set(tuple(sorted(e)) for e in G_train.edges())
    test_edges_set = set(tuple(sorted(e)) for e in G_test.edges())
    positive_edges = list(test_edges_set - train_edges_set)

    empty_metrics_keys = [f"{m}@{k}" for m in
                           ["precision", "recall", "f1", "accuracy", "hitrate", "ndcg", "diversity", "coverage"]
                           for k in topk_list]

    if not positive_edges:
        print("No positive (test) edges available. Skipping evaluation.")
        base = {k: 0.0 for k in empty_metrics_keys}
        for k in topk_list:
            base[f"auc_roc@{k}"] = 0.0
            base[f"auc_pr@{k}"] = 0.0
        return base

    negatives_needed = len(positive_edges)
    negative_edges = set()
    tries, max_tries = 0, negatives_needed * 20
    while len(negative_edges) < negatives_needed and tries < max_tries:
        tries += 1
        u, v = random.sample(all_nodes, 2)
        if u == v:
            continue
        e = tuple(sorted((u, v)))
        if e in train_edges_set or e in test_edges_set:
            continue
        negative_edges.add(e)
    negative_edges = list(negative_edges)

    X_states, y_labels = [], []
    rec_lists = defaultdict(list)

    for (u, v) in positive_edges:
        if u not in G_train or v not in G_train:
            continue
        state_vec, _ = build_state_vector(G_train, u, v, feature_cache)
        X_states.append(state_vec)
        y_labels.append(1)
        rec_lists[u].append((v, None))

    for (u, v) in negative_edges:
        if u not in G_train or v not in G_train:
            continue
        state_vec, _ = build_state_vector(G_train, u, v, feature_cache)
        X_states.append(state_vec)
        y_labels.append(0)
        rec_lists[u].append((v, None))

    if not X_states:
        print("No samples available for evaluation.")
        base = {k: 0.0 for k in empty_metrics_keys}
        for k in topk_list:
            base[f"auc_roc@{k}"] = 0.0
            base[f"auc_pr@{k}"] = 0.0
        return base

    X_tensor = torch.from_numpy(np.array(X_states, dtype=np.float32)).to(device)
    with torch.no_grad():
        logits, _ = model(X_tensor)
        scores = torch.sigmoid(logits.view(-1)).cpu().numpy().tolist()

    idx = 0
    y_true_bin, y_score_bin = [], []
    for (u, v), label in zip(positive_edges + negative_edges, y_labels):
        score = scores[idx]
        idx += 1
        y_true_bin.append(label)
        y_score_bin.append(score)
        lst = rec_lists[u]
        for i in range(len(lst)):
            if lst[i][0] == v and lst[i][1] is None:
                lst[i] = (v, score)
                break

    metrics = {}
    all_items = set(all_nodes)
    y_true_by_k = {k: [] for k in topk_list}
    y_score_by_k = {k: [] for k in topk_list}

    for k in topk_list:
        precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
        hitrate_list, ndcg_list, diversity_list = [], [], []
        rec_lists_for_coverage = []

        for u, lst in rec_lists.items():
            lst_sorted = sorted(lst, key=lambda x: x[1] if x[1] is not None else 0.0, reverse=True)
            recommended = [v for (v, s) in lst_sorted]

            gt_for_u = {y for (x, y) in positive_edges if x == u} | {x for (x, y) in positive_edges if y == u}
            if not gt_for_u:
                continue

            rec_lists_for_coverage.append(recommended)
            topk = recommended[:k]
            y_true_u = [1 if v in gt_for_u else 0 for v in topk]
            y_pred_u = [1] * len(topk)
            if not y_true_u:
                continue

            scores_u_topk = [s for (_, s) in lst_sorted[:k]]
            y_true_by_k[k].extend(y_true_u)
            y_score_by_k[k].extend(scores_u_topk)

            precision_list.append(precision_score(y_true_u, y_pred_u, zero_division=0))
            recall_list.append(recall_score(y_true_u, y_pred_u, zero_division=0))
            f1_list.append(f1_score(y_true_u, y_pred_u, zero_division=0))
            accuracy_list.append(accuracy_score(y_true_u, y_pred_u))
            hitrate_list.append(hit_rate_at_k(recommended, gt_for_u, k))
            ndcg_list.append(ndcg_at_k(recommended, gt_for_u, k))
            diversity_list.append(diversity_at_k(G_train, recommended, k))

        if not precision_list:
            for m in ["precision", "recall", "f1", "accuracy", "hitrate", "ndcg", "diversity", "coverage"]:
                metrics[f"{m}@{k}"] = 0.0
        else:
            metrics[f"precision@{k}"] = float(np.mean(precision_list))
            metrics[f"recall@{k}"] = float(np.mean(recall_list))
            metrics[f"f1@{k}"] = float(np.mean(f1_list))
            metrics[f"accuracy@{k}"] = float(np.mean(accuracy_list))
            metrics[f"hitrate@{k}"] = float(np.mean(hitrate_list))
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_list))
            metrics[f"diversity@{k}"] = float(np.mean(diversity_list))
            metrics[f"coverage@{k}"] = coverage_at_k(all_items, rec_lists_for_coverage, k)

    for k in topk_list:
        y_t, y_s = y_true_by_k[k], y_score_by_k[k]
        if not y_t or len(set(y_t)) < 2:
            metrics[f"auc_roc@{k}"] = 0.5
            metrics[f"auc_pr@{k}"] = 0.0
        else:
            try:
                metrics[f"auc_roc@{k}"] = roc_auc_score(y_t, y_s)
            except Exception:
                metrics[f"auc_roc@{k}"] = 0.5
            try:
                metrics[f"auc_pr@{k}"] = average_precision_score(y_t, y_s)
            except Exception:
                metrics[f"auc_pr@{k}"] = 0.0

    print("Evaluation metrics:", metrics)
    return metrics


############################################
# Part 9: Metric / reward / value / advantage plots
############################################

def plot_metrics_for_single_k(k_value, episodes, metrics_history, save_dir, episode_rewards=None, episode_values=None, episode_advantages=None):
    """Plots each metric in its true chronological (evaluation-checkpoint)
    order — not sorted — so the plot reflects the actual training dynamics,
    including any noise or instability."""
    if not episodes:
        print(f"No episodes recorded for k={k_value}; skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)
    metric_names = ["precision", "recall", "f1", "accuracy", "hitrate", "ndcg", "diversity", "coverage"]

    for met in metric_names:
        key = f"{met}@{k_value}"
        if key not in metrics_history or not metrics_history[key]:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(episodes, metrics_history[key], marker='o', label=f"{met}@{k_value}")
        plt.xlabel("Episode")
        plt.ylabel(met)
        plt.title(f"{met} over episodes (top@{k_value})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(save_dir, f"{met}_over_episodes_top@{k_value}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    for auc_key in (f"auc_roc@{k_value}", f"auc_pr@{k_value}"):
        if auc_key in metrics_history and metrics_history[auc_key]:
            label = auc_key.split("@")[0].upper().replace("_", "-")
            plt.figure(figsize=(8, 5))
            plt.plot(episodes, metrics_history[auc_key], marker='o', label=f"{label}@{k_value}")
            plt.xlabel("Episode")
            plt.ylabel(label)
            plt.title(f"{label} over episodes (top@{k_value})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            filename = os.path.join(save_dir, f"{auc_key.replace('@', '_over_episodes_top@')}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Saved: {filename}")

    for values, ylabel in [(episode_rewards, "Reward"), (episode_values, "Value"), (episode_advantages, "Advantage")]:
        if values:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(values) + 1), values, marker='o')
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} per Episode (top@{k_value})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = os.path.join(save_dir, f"{ylabel.lower()}_over_episodes_top@{k_value}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            print(f"Saved: {filename}")


############################################
# Part 10: Feature-weight evolution over episodes
############################################

def plot_feature_weights_evolution(k_value, episodes, feature_weights_history, save_dir):
    if not episodes or not feature_weights_history:
        print(f"No weight history to plot for k={k_value}.")
        return

    os.makedirs(save_dir, exist_ok=True)
    fw_array = np.vstack(feature_weights_history)
    feature_names_weighted = ["degree_u", "closeness_u", "betweenness_u", "pagerank_u", "ns_u"]

    plt.figure(figsize=(8, 5))
    for i, fname in enumerate(feature_names_weighted):
        plt.plot(episodes, fw_array[:, i], marker='o', label=fname)
    plt.xlabel("Episode")
    plt.ylabel("Weight value")
    plt.title(f"Feature Weights Evolution Over Episodes (top@{k_value})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(save_dir, f"feature_weights_evolution_top@{k_value}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved feature weights evolution plot to: {filename}")


############################################
# Part 11: Feature importance bar plot
############################################

def analyze_and_plot_feature_importance(model, save_dir, suffix=""):
    os.makedirs(save_dir, exist_ok=True)
    feature_names_weighted = ["degree_u", "closeness_u", "betweenness_u", "pagerank_u", "ns_u"]

    with torch.no_grad():
        weights = (1.0 + model.feature_weights_raw).detach().cpu().numpy()
        biases = model.feature_bias.detach().cpu().numpy()

    print("\n=== Feature Weights (final w_i = 1 + w_raw) for Structural Features of u ===")
    for name, w, b in zip(feature_names_weighted, weights, biases):
        print(f"{name:15s}  weight = {w:.4f}   bias = {b:.4f}")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(feature_names_weighted))
    plt.bar(x, np.abs(weights), tick_label=feature_names_weighted)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("|weight|")
    plt.title("Feature Importance (Learned Weights on u's Structural Features)")
    plt.tight_layout()

    fname = f"feature_importance_weights_{suffix}.png" if suffix else "feature_importance_weights.png"
    filename = os.path.join(save_dir, fname)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {filename}")


############################################
# Part 12: Combined plot across all k values
############################################

def plot_all_k_metrics(all_metrics_history, all_episodes_recorded, ks, save_dir):
    """Plots each metric's true chronological values across evaluation
    checkpoints, for all top-k scenarios in the same figure. No noise is
    added and values are not sorted — the plot reflects the model's actual
    measured performance."""
    os.makedirs(save_dir, exist_ok=True)
    combined_metrics = ["precision", "recall", "f1", "accuracy", "hitrate", "ndcg", "auc_roc", "auc_pr"]

    for met in combined_metrics:
        plt.figure(figsize=(8, 5))
        any_plotted = False

        for k in ks:
            metrics_history = all_metrics_history.get(k, {})
            episodes = all_episodes_recorded.get(k, [])
            key = f"{met}@{k}"
            if not episodes or key not in metrics_history or not metrics_history[key]:
                continue

            y = np.array(metrics_history[key], dtype=float)
            plt.plot(episodes, y, marker='o', label=f"{met}@{k}")
            any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        plt.xlabel("Episode")
        plt.ylabel(met)
        plt.title(f"{met} over episodes (top@{','.join(str(k) for k in ks)})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(save_dir, f"{met}_over_episodes_all_k.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved combined metric plot: {filename}")


############################################
# Part 13: CLI / main
############################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWAC-style Actor-Critic friend recommendation")
    parser.add_argument("--graph", default="data/network.txt", help="Path to edge-list graph file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved plots")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--l1-lambda", type=float, default=1e-4)
    parser.add_argument("--ns-reg-alpha", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    G = load_graph_from_file(args.graph)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ks = args.topk
    all_metrics_history = {}
    all_episodes_recorded = {}

    for k in ks:
        print(f"\n================= Training for top@{k} =================")
        model, metrics_history, fw_hist, eps_rec, ep_rewards, ep_values, ep_advs = train_actor_critic(
            G, num_episodes=args.episodes, learning_rate=args.lr, gamma=args.gamma,
            topk_list=(k,), eval_every=args.eval_every, device=device,
            l1_lambda=args.l1_lambda, ns_reg_alpha=args.ns_reg_alpha,
        )
        print(f"Training finished for top@{k}.")

        all_metrics_history[k] = metrics_history
        all_episodes_recorded[k] = eps_rec

        plot_metrics_for_single_k(
            k_value=k, episodes=eps_rec, metrics_history=metrics_history, save_dir=args.output_dir,
            episode_rewards=ep_rewards, episode_values=ep_values, episode_advantages=ep_advs,
        )
        plot_feature_weights_evolution(k_value=k, episodes=eps_rec, feature_weights_history=fw_hist, save_dir=args.output_dir)
        analyze_and_plot_feature_importance(model, save_dir=args.output_dir, suffix=f"top@{k}")

    plot_all_k_metrics(all_metrics_history=all_metrics_history, all_episodes_recorded=all_episodes_recorded, ks=ks, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
