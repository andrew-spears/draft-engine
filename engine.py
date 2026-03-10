"""
Expectimax search engine.

Tree alternates chance nodes (random draw, averaged) and decision nodes
(pick bundle, maxed). Heuristic at leaves is the current score via lookup table.
"""

import numpy as np
import numba
from numba import int64, float64
from game import GameConfig, total_score_from_table, generate_assignments


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


class Engine:
    def __init__(self, depth, fanout, config=None):
        if config is None:
            config = GameConfig()
        self.depth = depth
        self.fanout = fanout
        self.config = config
        self.score_table = config.make_score_table()
        self.node_count = 0

    def search_value(self, stashed, remaining):
        cfg = self.config
        deck_size = sum(remaining)
        if self.depth == 0 or deck_size < cfg.draw_size:
            self.node_count += 1
            return total_score_from_table(
                np.array(stashed, dtype=np.int64), self.score_table)

        val, nodes = _search(
            np.array(stashed, dtype=np.int64),
            np.array(remaining, dtype=np.int64),
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
