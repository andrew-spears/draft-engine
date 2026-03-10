"""
Game: one player collects goods from a shared pool.

A pool of goods has known quantities per type (e.g. 10 type-0, 9 type-1, ...).
Each round, goods are drawn from the pool and each good is randomly assigned
to `overlap_degree` of the `num_bundles` bundles. The player picks one bundle
to keep; unpicked goods are discarded. Repeat until the pool is exhausted.
Score based on collected counts.
"""

import numpy as np
import numba
from dataclasses import dataclass, field
from typing import Callable


# --- Scoring ---

def face_score(type_value, count):
    """Default scoring: paired evens good, odds bad.
    type_value is 1-indexed face value."""
    if count == 0:
        return 0
    if count == 2:
        return 2 * type_value
    if count == 4:
        return 8 * type_value
    if count == 8:
        return 16 * type_value
    return -type_value * count


# --- Config ---

@dataclass
class GameConfig:
    num_types: int = 10
    init_pool: tuple = tuple(range(20, 10, -1))  # type i has 20-i goods initially
    draw_size: int = 10                           # goods drawn per round
    num_bundles: int = 5                         # choices per round
    overlap_degree: int = 2                      # bundles each good appears in
    score_fn: Callable = field(default=face_score)

    @property
    def num_rounds(self):
        return sum(self.init_pool) // self.draw_size

    @property
    def init_stashed(self):
        return (0,) * self.num_types

    def make_score_table(self, max_count=None):
        """Build score_table[type_idx, count] as a 2D numpy array.
        type_idx is 0-based; score_fn receives 1-indexed type_value."""
        if max_count is None:
            max_count = max(self.init_pool) + self.draw_size
        table = np.zeros((self.num_types, max_count + 1), dtype=np.float64)
        for i in range(self.num_types):
            for c in range(max_count + 1):
                table[i, c] = self.score_fn(i + 1, c)
        return table


# --- Numba scoring from table ---

@numba.jit(nopython=True, cache=True)
def total_score_from_table(stashed, score_table):
    total = 0.0
    for i in range(stashed.shape[0]):
        total += score_table[i, stashed[i]]
    return total


# --- Assignment generation ---

@numba.jit(nopython=True, cache=True)
def generate_assignments(draw_size, num_bundles, overlap_degree):
    """For each drawn good, pick `overlap_degree` random bundles.
    Returns assignments array of shape (draw_size, overlap_degree)."""
    assignments = np.empty((draw_size, overlap_degree), dtype=numba.int64)
    # temp array for Fisher-Yates partial shuffle
    perm = np.empty(num_bundles, dtype=numba.int64)
    for j in range(draw_size):
        for i in range(num_bundles):
            perm[i] = i
        for i in range(overlap_degree):
            swap_idx = np.random.randint(i, num_bundles)
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
            assignments[j, i] = perm[i]
    return assignments


# --- Sampling (numpy, used for the outer game loop) ---

_rng = np.random.default_rng()


def sample_transitions(stashed, remaining, config):
    """Sample one draw and return (transitions, draw, assignments).
    draw is 0-indexed type indices."""
    num_types = config.num_types
    draw_size = config.draw_size
    num_bundles = config.num_bundles
    overlap_degree = config.overlap_degree

    rem_arr = np.array(remaining, dtype=np.int64)
    counts = _rng.multivariate_hypergeometric(rem_arr, draw_size)
    draw = np.repeat(np.arange(num_types, dtype=np.int64), counts)
    _rng.shuffle(draw)

    # new remaining after removing all drawn goods
    new_remaining = tuple(int(remaining[i] - counts[i]) for i in range(num_types))

    # random bundle assignments
    assignments = generate_assignments(draw_size, num_bundles, overlap_degree)

    # build each bundle's transition
    transitions = []
    for b in range(num_bundles):
        s = list(stashed)
        for j in range(draw_size):
            for r in range(overlap_degree):
                if assignments[j, r] == b:
                    s[draw[j]] += 1
                    break
        transitions.append((tuple(s), new_remaining))

    return transitions, draw, assignments
