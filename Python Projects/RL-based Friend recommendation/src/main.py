# -*- coding: utf-8 -*-
"""
RL-Based Friend Recommendation (Deep Q-Learning)
==================================================
Trains a DQN agent to recommend friends within a social network graph,
using network centrality measures (degree, closeness, betweenness) to
shape the reward signal. Evaluated across multiple epsilon (exploration
rate) scenarios on a held-out set of test users.

Usage
-----
    python src/main.py --graph data/dolphins.gml --episodes 2000

Note
----
This project was originally developed in a "vibe coding" style (rapid,
exploratory, AI-assisted iteration) rather than a fully test-driven
design. See the README for known limitations before using it as a
reference implementation.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# ==========================================
# Graph loading & centrality
# ==========================================
def load_graph(filepath: str) -> nx.Graph:
    """Load a graph from either GML (.gml) or a plain edge-list text file
    (.txt / .edgelist / .csv — one 'source target' pair per line)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".gml":
        return nx.read_gml(filepath)
    # Default: treat as whitespace/comma-separated edge list.
    try:
        return nx.read_edgelist(filepath)
    except Exception:
        # Fallback for comma-separated edge lists.
        return nx.read_edgelist(filepath, delimiter=",")


def calculate_centrality_measures(G: nx.Graph) -> dict:
    """Return {node: (degree_centrality, closeness, betweenness, modularity)}.

    `modularity` is a per-node proxy: the fraction of each node's edges that
    stay within its own detected community (via greedy modularity
    communities). This is a simplification, not the formal per-node
    contribution to Newman's modularity Q, but unlike the previous
    always-0.0 placeholder it actually varies with network structure.
    """
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node_to_community = {node: idx for idx, comm in enumerate(communities) for node in comm}

    modularity = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            modularity[node] = 0.0
            continue
        own_community = node_to_community[node]
        same_community_count = sum(1 for n in neighbors if node_to_community[n] == own_community)
        modularity[node] = same_community_count / len(neighbors)

    return {node: (degree[node], closeness[node], betweenness[node], modularity[node]) for node in G.nodes()}


# ==========================================
# Environment
# ==========================================
class FriendRecommendationEnv:
    def __init__(self, graph: nx.Graph, centrality: dict):
        self.graph = graph
        self.centrality = centrality
        self.node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.users = list(self.node_to_idx.keys())
        self.current_user = None
        self.friend_rings = {node: set(graph.neighbors(node)) for node in graph.nodes()}
        self.friend_ring_metrics = {user: self._calculate_friend_ring_metrics(user) for user in self.users}

    def reset(self):
        self.current_user = random.choice(self.users)
        return self._get_state()

    def _get_state(self):
        state = np.zeros(len(self.graph.nodes()), dtype=np.float32)
        state[self.node_to_idx[self.current_user]] = 1.0
        for friend in self.friend_rings[self.current_user]:
            state[self.node_to_idx[friend]] = 1.0
        return state

    def step(self, action_idx: int):
        recommended_user = self.idx_to_node[action_idx]
        if recommended_user not in self.graph.nodes():
            raise KeyError(f"Recommended user '{recommended_user}' not in graph nodes.")

        reward = self._calculate_reward(recommended_user)

        self.friend_rings[self.current_user].add(recommended_user)
        self.graph.add_edge(self.current_user, recommended_user)
        self.friend_ring_metrics[self.current_user] = self._calculate_friend_ring_metrics(self.current_user)

        return self._get_state(), reward, True

    def _calculate_friend_ring_metrics(self, user):
        friends = self.friend_rings[user]
        if not friends:
            return {"centrality": 0.0, "closeness": 0.0, "betweenness": 0.0, "modularity": 0.0}

        centrality = np.mean([self.centrality[f][0] for f in friends])
        closeness = np.mean([self.centrality[f][1] for f in friends])
        betweenness = np.mean([self.centrality[f][2] for f in friends])
        modularity = np.mean([self.centrality[f][3] for f in friends])
        return {"centrality": centrality, "closeness": closeness, "betweenness": betweenness, "modularity": modularity}

    def _calculate_reward(self, recommended_user):
        if recommended_user == self.current_user:
            return -3  # Penalty for recommending oneself

        current_metrics = self.friend_ring_metrics[self.current_user]

        self.friend_rings[self.current_user].add(recommended_user)
        new_metrics = self._calculate_friend_ring_metrics(self.current_user)
        self.friend_rings[self.current_user].remove(recommended_user)

        changes = {m: new_metrics[m] - current_metrics[m] for m in current_metrics}
        positive_changes = sum(1 for c in changes.values() if c > 0)
        negative_changes = sum(1 for c in changes.values() if c < 0)

        if positive_changes == 4:
            return 1.0  # All metrics improved
        elif negative_changes == 4:
            return -1.0  # All metrics worsened
        else:
            return 0.5  # Mixed result — partial improvement, distinct from a full win


# ==========================================
# DQN model
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# Evaluation
# ==========================================
def evaluate_model(env: FriendRecommendationEnv, q_network: DQN, test_users, graph: nx.Graph):
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    for user_i in test_users:
        neighbors_i = set(graph.neighbors(user_i))
        state = env.reset()
        state[env.node_to_idx[user_i]] = 1.0
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_idx = torch.argmax(q_network(state)).item()
        recommended_user = env.idx_to_node[action_idx]

        if recommended_user in neighbors_i:
            tp, fp = 1, 0
            fn = len(neighbors_i - {recommended_user})
        else:
            tp, fp = 0, 1
            fn = len(neighbors_i)
        tn = len(env.users) - tp - fp - fn

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        tn_list.append(tn)

    return tp_list, fp_list, fn_list, tn_list


def calculate_metrics(tp, fp, fn, tn):
    """Standard precision/recall/accuracy/F1 from confusion-matrix counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1


# ==========================================
# Training loop
# ==========================================
def train_and_evaluate(env, graph, test_users, episodes, gamma, epsilon_values, epsilon_decay, lr, interval):
    metrics_per_scenario = {eps: [] for eps in epsilon_values}
    reward_per_scenario = {eps: [] for eps in epsilon_values}
    q_values_per_scenario = {eps: [] for eps in epsilon_values}

    for initial_epsilon in epsilon_values:
        print(f"Training with epsilon={initial_epsilon}")
        epsilon = initial_epsilon

        state_size = len(env.users)
        action_size = len(env.users)
        q_network = DQN(state_size, action_size)
        optimizer = optim.Adam(q_network.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
        reward_list, q_value_list = [], []

        for episode in range(episodes):
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            total_reward = 0
            q_values_episode = []
            done = False
            step_count = 0

            while not done and step_count < 10:
                if random.random() < epsilon:
                    action_idx = random.randint(0, action_size - 1)
                else:
                    with torch.no_grad():
                        q_values = q_network(state)
                        action_idx = torch.argmax(q_values).item()
                        q_values_episode.append(q_values[0, action_idx].item())

                try:
                    next_state, reward, done = env.step(action_idx)
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                except KeyError:
                    reward = -1.0
                    done = True

                target = torch.tensor(
                    reward + (gamma * torch.max(q_network(next_state)).item() if not done else 0),
                    dtype=torch.float32,
                )
                prediction = q_network(state)[0, action_idx]  # 0-dim scalar, matches target's shape

                loss = loss_fn(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_reward += reward
                state = next_state
                step_count += 1

            reward_per_scenario[initial_epsilon].append(total_reward)
            reward_list.append(total_reward)
            mean_q = np.mean(q_values_episode) if q_values_episode else 0.0
            q_values_per_scenario[initial_epsilon].append(mean_q)
            q_value_list.append(mean_q)

            epsilon *= epsilon_decay

            if (episode + 1) % interval == 0:
                tp, fp, fn, tn = evaluate_model(env, q_network, test_users, graph)
                accuracy, precision, recall, f1 = calculate_metrics(sum(tp), sum(fp), sum(fn), sum(tn))
                metrics_per_scenario[initial_epsilon].append((accuracy, precision, recall, f1))
                accuracy_list.append(accuracy)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                print(
                    f"Episode {episode + 1}, epsilon={epsilon:.4f}: "
                    f"Acc={accuracy:.2f}, Prec={precision:.2f}, Rec={recall:.2f}, F1={f1:.2f}"
                )

        print(f"\nFinal Averages for Epsilon={initial_epsilon}:")
        print(f"  Average Accuracy: {np.mean(accuracy_list) if accuracy_list else 0.0:.2f}")
        print(f"  Average Precision: {np.mean(precision_list) if precision_list else 0.0:.2f}")
        print(f"  Average Recall: {np.mean(recall_list) if recall_list else 0.0:.2f}")
        print(f"  Average F1: {np.mean(f1_list) if f1_list else 0.0:.2f}")
        print(f"  Average Reward: {np.mean(reward_list) if reward_list else 0.0:.2f}")
        print(f"  Average Q-Value: {np.mean(q_value_list) if q_value_list else 0.0:.2f}\n")

    return metrics_per_scenario, reward_per_scenario, q_values_per_scenario


# ==========================================
# Plotting (saved to disk instead of plt.show())
# ==========================================
def plot_q_and_reward(reward_per_scenario, q_values_per_scenario, episodes, epsilon_values, output_dir):
    x_points = list(range(1, episodes + 1))
    fig, axs = plt.subplots(len(epsilon_values), 1, figsize=(12, len(epsilon_values) * 4))
    fig.suptitle("Q-Values Across Episodes", fontsize=16)

    for idx, epsilon in enumerate(epsilon_values):
        ax = axs[idx]
        q_values = q_values_per_scenario[epsilon]
        ax.plot(x_points, q_values, label="Q-Value", color="orange", marker="x", markersize=2, linewidth=0.7)
        ax.set_title(f"Epsilon={epsilon}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "q_values_across_episodes.png"), dpi=300)
    plt.close()


def plot_metrics(metrics_per_scenario, reward_per_scenario, episodes, interval, epsilon_values, output_dir):
    x_points_intervals = list(range(interval, episodes + 1, interval))
    x_points_rewards = list(range(1, episodes + 1))

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        for epsilon, results in metrics_per_scenario.items():
            y_points = sorted(result[i] for result in results)
            plt.plot(x_points_intervals, y_points, label=f"Epsilon={epsilon}")
        plt.title(f"{metric} Across Episodes")
        plt.xlabel("Episodes")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{metric.lower().replace('-', '_')}.png"), dpi=300)
        plt.close()

    fig, axs = plt.subplots(len(epsilon_values), 1, figsize=(10, len(epsilon_values) * 4))
    fig.suptitle("Reward Trends Per Episode for Different Epsilon Scenarios", fontsize=16)
    for idx, epsilon in enumerate(epsilon_values):
        ax = axs[idx]
        rewards = reward_per_scenario[epsilon]
        ax.plot(x_points_rewards, rewards, label=f"Epsilon={epsilon}", color="blue", marker="o", markersize=2, linewidth=0.7)
        ax.set_title(f"Rewards for Epsilon={epsilon}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "reward_trends.png"), dpi=300)
    plt.close()


# ==========================================
# CLI / main
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL-based friend recommendation (DQN)")
    parser.add_argument("--graph", default="data/dolphins.txt", help="Path to input graph file (.gml or edge-list .txt)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved plots")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=100, help="Evaluation interval (in episodes)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    G = load_graph(args.graph)
    centrality = calculate_centrality_measures(G)
    env = FriendRecommendationEnv(G, centrality)

    users = list(G.nodes())
    train_users, test_users = train_test_split(users, test_size=args.test_size)

    metrics_per_scenario, reward_per_scenario, q_values_per_scenario = train_and_evaluate(
        env, G, test_users, args.episodes, args.gamma,
        args.epsilon_values, args.epsilon_decay, args.lr, args.interval,
    )

    plot_metrics(metrics_per_scenario, reward_per_scenario, args.episodes, args.interval, args.epsilon_values, args.output_dir)
    plot_q_and_reward(reward_per_scenario, q_values_per_scenario, args.episodes, args.epsilon_values, args.output_dir)


if __name__ == "__main__":
    main()
