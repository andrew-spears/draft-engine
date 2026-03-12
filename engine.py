"""
Expectimax search engine.

Tree alternates chance nodes (random draw, averaged) and decision nodes
(pick bundle, maxed). Heuristic at leaves is the current score via lookup table.

Supports pluggable leaf evaluation via `leaf_fn` for batched NN evaluation.
"""

import numpy as np
import numba
from numba import int64, float64
from game import GameConfig, total_score_from_table, generate_assignments, sample_transitions


@numba.jit(nopython=True, cache=True)
def sample_draw(remaining, deck_size, draw_size, num_types):
    """Sample draw_size goods without replacement via cumulative sum lookup.
    Modifies remaining in place — caller must copy first."""
    draw = np.empty(draw_size, dtype=int64)
    size = deck_size
    for j in range(draw_size):
        r = np.random.randint(0, size)
        cumsum = 0
        for i in range(num_types):
            cumsum += remaining[i]
            if r < cumsum:
                draw[j] = i
                remaining[i] -= 1
                size -= 1
                break
    return draw


@numba.jit(nopython=True, cache=True)
def _copy_array(src, n):
    dst = np.empty(n, dtype=int64)
    for i in range(n):
        dst[i] = src[i]
    return dst


@numba.jit(nopython=True, cache=True)
def _apply_bundle(stashed, draw, assignments, bundle_idx, draw_size, overlap_degree, num_types):
    """Copy stashed and add goods assigned to bundle_idx."""
    new = np.empty(num_types, dtype=int64)
    for i in range(num_types):
        new[i] = stashed[i]
    for j in range(draw_size):
        for r in range(overlap_degree):
            if assignments[j, r] == bundle_idx:
                new[draw[j]] += 1
                break
    return new


@numba.jit(nopython=True, cache=True)
def _search(stashed, remaining, deck_size, depth, fanout,
            num_types, draw_size, num_bundles, overlap_degree, score_table):
    """Expectimax search. Returns (value, nodes_visited)."""
    if depth == 0 or deck_size < draw_size:
        return total_score_from_table(stashed, score_table), 1

    sum_value = 0.0
    total_nodes = 0

    for _ in range(fanout):
        rem_copy = _copy_array(remaining, num_types)
        draw = sample_draw(rem_copy, deck_size, draw_size, num_types)
        assignments = generate_assignments(draw_size, num_bundles, overlap_degree)

        best = -1e9
        for b in range(num_bundles):
            new_stash = _apply_bundle(stashed, draw, assignments, b,
                                      draw_size, overlap_degree, num_types)
            val, nodes = _search(new_stash, rem_copy, deck_size - draw_size,
                                 depth - 1, fanout,
                                 num_types, draw_size, num_bundles, overlap_degree, score_table)
            total_nodes += nodes
            if val > best:
                best = val

        sum_value += best

    return sum_value / fanout, total_nodes


def expand_tree(stashed, remaining, depth, fanout, config, score_table):
    """Expand expectimax tree in Python, collecting leaves for batched evaluation.

    Returns (leaves, backup_fn):
      - leaves: list of (stashed, remaining, is_terminal) tuples
      - backup_fn(values): takes a list/array of float values (one per leaf),
        propagates max-over-bundles then mean-over-fanout back to root, returns scalar.
    """
    leaves = []

    def _expand(stashed, remaining, depth):
        deck_size = sum(remaining)
        is_terminal = deck_size < config.draw_size

        if depth == 0 or is_terminal:
            idx = len(leaves)
            leaves.append((stashed, remaining, is_terminal))
            return lambda values: values[idx]

        # Chance node: sample `fanout` draws, average results
        sample_backups = []
        for _ in range(fanout):
            transitions, _, _ = sample_transitions(stashed, remaining, config)

            # Decision node: max over bundles
            bundle_backups = []
            for s, r in transitions:
                bundle_backups.append(_expand(s, r, depth - 1))

            # Capture bundle_backups in closure
            def _max_backup(values, _bbs=bundle_backups):
                best = -1e9
                for bb in _bbs:
                    v = bb(values)
                    if v > best:
                        best = v
                return best

            sample_backups.append(_max_backup)

        def _mean_backup(values, _sbs=sample_backups):
            total = 0.0
            for sb in _sbs:
                total += sb(values)
            return total / len(_sbs)

        return _mean_backup

    backup_fn = _expand(stashed, remaining, depth)
    return leaves, backup_fn


class Engine:
    def __init__(self, depth, fanout, config=None, leaf_fn=None):
        """
        leaf_fn: optional callable that takes a list of (stashed, remaining, is_terminal)
                 tuples and returns a list of float values. If provided, uses
                 expand_tree + leaf_fn for batched evaluation instead of numba _search.
        """
        if config is None:
            config = GameConfig()
        self.depth = depth
        self.fanout = fanout
        self.config = config
        self.score_table = config.make_score_table()
        self.node_count = 0
        self.leaf_fn = leaf_fn

    def search_value(self, stashed, remaining):
        cfg = self.config
        deck_size = sum(remaining)

        if self.leaf_fn is not None:
            # Batched leaf evaluation path
            if self.depth == 0 or deck_size < cfg.draw_size:
                self.node_count += 1
                # Single leaf — just call leaf_fn directly
                is_terminal = deck_size < cfg.draw_size
                values = self.leaf_fn([(stashed, remaining, is_terminal)])
                return values[0]

            leaves, backup_fn = expand_tree(
                stashed, remaining, self.depth, self.fanout, cfg, self.score_table
            )
            self.node_count += len(leaves)
            values = self.leaf_fn(leaves)
            return backup_fn(values)
        else:
            # Numba path
            stashed_arr = np.array(stashed, dtype=np.int64)
            remaining_arr = np.array(remaining, dtype=np.int64)

            if self.depth == 0 or deck_size < cfg.draw_size:
                self.node_count += 1
                return total_score_from_table(stashed_arr, self.score_table)

            val, nodes = _search(
                stashed_arr, remaining_arr,
                deck_size, self.depth, self.fanout,
                cfg.num_types, cfg.draw_size, cfg.num_bundles,
                cfg.overlap_degree, self.score_table,
            )
            self.node_count += nodes
            return val

    def get_action(self, transitions):
        best_action = 0
        best_value = -1e9
        for idx, (stashed, remaining) in enumerate(transitions):
            v = self.search_value(stashed, remaining)
            if v > best_value:
                best_value = v
                best_action = idx
        return best_action
