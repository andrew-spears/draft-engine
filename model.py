"""
Neural network value function for the collection game.

Approximates expected score under optimal play for a given game state.
"""

import torch
import torch.nn as nn
import numpy as np
from game import GameConfig, total_score_from_table


class ValueNet(nn.Module):
    """Small fully-connected network: state -> predicted value."""

    def __init__(self, num_types, hidden_size=32, num_layers=3):
        super().__init__()
        input_size = 2 * num_types  # stashed + remaining, both normalized
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_state(stashed, remaining, config):
    """Encode a game state as a normalized float tensor.

    stashed/remaining: tuples of ints, length num_types.
    Returns: tensor of shape (2 * num_types,).
    """
    pool = np.array(config.init_pool, dtype=np.float32)
    # Avoid division by zero for types with 0 in init_pool
    pool = np.maximum(pool, 1.0)
    s = np.array(stashed, dtype=np.float32) / pool
    r = np.array(remaining, dtype=np.float32) / pool
    return torch.tensor(np.concatenate([s, r]), dtype=torch.float32)


def encode_states_batch(states, config):
    """Encode multiple (stashed, remaining) pairs as a batch tensor.

    states: list of (stashed, remaining) tuples.
    Returns: tensor of shape (len(states), 2 * num_types).
    """
    pool = np.array(config.init_pool, dtype=np.float32)
    pool = np.maximum(pool, 1.0)
    rows = []
    for stashed, remaining in states:
        s = np.array(stashed, dtype=np.float32) / pool
        r = np.array(remaining, dtype=np.float32) / pool
        rows.append(np.concatenate([s, r]))
    return torch.tensor(np.array(rows), dtype=torch.float32)


def make_leaf_fn(model, config):
    """Create a leaf_fn that uses the NN for non-terminal states.

    Returns a callable: list of (stashed, remaining, is_terminal) -> list of floats.
    Terminal states are evaluated with score_table; non-terminal states get a
    single batched forward pass through the model.
    """
    score_table = config.make_score_table()

    def leaf_fn(leaves):
        n = len(leaves)
        values = [0.0] * n
        nn_indices = []
        nn_states = []

        for i, (stashed, remaining, is_terminal) in enumerate(leaves):
            if is_terminal:
                stashed_arr = np.array(stashed, dtype=np.int64)
                values[i] = float(total_score_from_table(stashed_arr, score_table))
            else:
                nn_indices.append(i)
                nn_states.append((stashed, remaining))

        if nn_states:
            X = encode_states_batch(nn_states, config)
            with torch.no_grad():
                preds = model(X).cpu().numpy()
            for j, idx in enumerate(nn_indices):
                values[idx] = float(preds[j])

        return values

    return leaf_fn


def make_heuristic_leaf_fn(config):
    """Create a leaf_fn that uses score_table for all states (terminal or not).

    Useful as a baseline: equivalent logic to the numba _search path but
    executed through the batched expand_tree infrastructure.
    """
    score_table = config.make_score_table()

    def leaf_fn(leaves):
        values = []
        for stashed, remaining, is_terminal in leaves:
            stashed_arr = np.array(stashed, dtype=np.int64)
            values.append(float(total_score_from_table(stashed_arr, score_table)))
        return values

    return leaf_fn


def nn_get_action(model, transitions, config):
    """Pick best action using NN value predictions (no search)."""
    X = encode_states_batch(transitions, config)
    with torch.no_grad():
        values = model(X)
    return values.argmax().item()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, config, hidden_size=32, num_layers=3):
    model = ValueNet(config.num_types, hidden_size, num_layers)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
