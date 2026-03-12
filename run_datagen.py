"""
Parallel data generation. Saves to .npz for later training.

Usage:
    python run_datagen.py                              # 5000 games, half of cores
    python run_datagen.py --games 50000 --workers 64   # big run
    python run_datagen.py --depth 3 --fanout 20        # deeper search
    python run_datagen.py --out data/run1.npz          # custom output path

Output .npz contains:
    stashed:   (N, num_types) int array
    remaining: (N, num_types) int array
    values:    (N,) float array
"""

import argparse
import multiprocessing as mp
import os
import time
import numpy as np
from game import GameConfig, face_score, sample_transitions
from engine import Engine


def make_config():
    return GameConfig(
        num_types=5,
        init_pool=(8, 7, 6, 5, 4),
        draw_size=5,
        num_bundles=4,
        overlap_degree=2,
        score_fn=face_score,
    )


def worker_generate(args):
    """Worker function: generate data for a chunk of games."""
    worker_id, num_games, depth, fanout = args
    config = make_config()
    engine = Engine(depth, fanout, config)

    stashed_list = []
    remaining_list = []
    values_list = []

    for game_i in range(num_games):
        if (game_i + 1) % max(1, num_games // 5) == 0:
            print(f"  Worker {worker_id}: game {game_i + 1}/{num_games}")

        stashed = config.init_stashed
        remaining = config.init_pool

        for _round in range(config.num_rounds):
            if sum(remaining) < config.draw_size:
                break

            transitions, _, _ = sample_transitions(stashed, remaining, config)

            best_value = -1e9
            best_idx = 0
            for idx, (s, r) in enumerate(transitions):
                v = engine.search_value(s, r)
                stashed_list.append(s)
                remaining_list.append(r)
                values_list.append(v)
                if v > best_value:
                    best_value = v
                    best_idx = idx

            stashed, remaining = transitions[best_idx]

    return stashed_list, remaining_list, values_list, engine.node_count


def main():
    p = argparse.ArgumentParser(description="Generate training data with parallel search")
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--fanout", type=int, default=20)
    p.add_argument("--workers", type=int, default=None, help="Defaults to half of cpu count")
    p.add_argument("--out", type=str, default="data.npz")
    args = p.parse_args()

    n_workers = args.workers or mp.cpu_count()//2
    games_per_worker = args.games // n_workers
    remainder = args.games % n_workers

    print(f"Generating {args.games} games across {n_workers} workers")
    print(f"Search: depth={args.depth}, fanout={args.fanout}")

    # Trigger numba compilation and benchmark a single search call
    print("Compiling numba functions...")
    config = make_config()
    engine = Engine(args.depth, args.fanout, config)
    engine.search_value(config.init_stashed, config.init_pool)
    # Benchmark: time several calls for a stable estimate
    t_bench = time.time()
    n_bench = 20
    for _ in range(n_bench):
        engine.search_value(config.init_stashed, config.init_pool)
    secs_per_call = (time.time() - t_bench) / n_bench

    # Estimate: each game has ~num_rounds rounds, each round evaluates num_bundles transitions
    rounds = config.num_rounds
    calls_per_game = rounds * config.num_bundles
    total_calls = args.games * calls_per_game
    est_serial = total_calls * secs_per_call
    est_parallel = est_serial / n_workers
    print(f"Benchmark: {secs_per_call*1000:.1f}ms/search call")
    print(f"Estimate: {args.games} games x ~{rounds} rounds x {config.num_bundles} bundles "
          f"= ~{total_calls:,} search calls")
    print(f"Estimated time: ~{est_parallel:.0f}s ({est_parallel/60:.1f}min) "
          f"across {n_workers} workers")

    worker_args = []
    for i in range(n_workers):
        n = games_per_worker + (1 if i < remainder else 0)
        if n > 0:
            worker_args.append((i, n, args.depth, args.fanout))

    t0 = time.time()
    with mp.Pool(len(worker_args)) as pool:
        results = pool.map(worker_generate, worker_args)
    elapsed = time.time() - t0

    # Merge results
    all_stashed = []
    all_remaining = []
    all_values = []
    total_nodes = 0
    for stashed_list, remaining_list, values_list, nodes in results:
        all_stashed.extend(stashed_list)
        all_remaining.extend(remaining_list)
        all_values.extend(values_list)
        total_nodes += nodes

    stashed_arr = np.array(all_stashed, dtype=np.int32)
    remaining_arr = np.array(all_remaining, dtype=np.int32)
    values_arr = np.array(all_values, dtype=np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, stashed=stashed_arr, remaining=remaining_arr, values=values_arr)

    print(f"\nSaved {len(all_values)} samples to {args.out}")
    print(f"Time: {elapsed:.1f}s ({args.games / elapsed:.0f} games/s)")
    print(f"Nodes searched: {total_nodes:,}")
    print(f"Value range: [{values_arr.min():.1f}, {values_arr.max():.1f}], "
          f"mean: {values_arr.mean():.1f}")


if __name__ == "__main__":
    main()
