import numpy as np
import time
from game import GameConfig, total_score_from_table, sample_transitions
from engine import Engine

config = GameConfig()
score_table = config.make_score_table()


def play_game(eng):
    stashed, remaining = config.init_stashed, config.init_pool
    for _ in range(config.num_rounds):
        transitions, _, _ = sample_transitions(stashed, remaining, config)
        stashed, remaining = transitions[eng.get_action(transitions)]
    return total_score_from_table(np.array(stashed, dtype=np.int64), score_table)


def play_game_random():
    stashed, remaining = config.init_stashed, config.init_pool
    for _ in range(config.num_rounds):
        transitions, _, _ = sample_transitions(stashed, remaining, config)
        stashed, remaining = transitions[np.random.randint(len(transitions))]
    return total_score_from_table(np.array(stashed, dtype=np.int64), score_table)


def run_benchmark(name, n, play_fn, eng=None):
    start = time.time()
    scores = [play_fn() for _ in range(n)]
    elapsed = time.time() - start

    nodes = eng.node_count if eng else 0
    nodes_per_move = nodes / n / config.num_rounds if nodes else 0
    us_per_node = elapsed / nodes * 1e6 if nodes else 0

    stats = f"{np.mean(scores):6.1f}  {np.std(scores):5.1f}  {elapsed/n:7.3f}s"
    node_stats = f"{nodes_per_move:12.0f}  {us_per_node:7.1f}" if nodes else f"{'--':>12s}  {'--':>8s}"
    print(f"{name:>25s}  {n:5d}  {stats}  {node_stats}")


def compute_fanout(depth, target_nodes_per_move, num_bundles):
    """Find fanout that gives approximately num_bundles*(num_bundles*fanout)^depth."""
    fanout = ((target_nodes_per_move / num_bundles) ** (1 / depth)) / num_bundles
    return max(round(fanout), 2)


# --- Warmup ---
print("Warming up numba...", end=" ", flush=True)
play_game(Engine(1, 5, config))
print("done.\n")

# --- Header ---
print(f"{'Strategy':>25s}  {'n':>5s}  {'mean':>6s}  {'std':>5s}  {'s/game':>8s}  {'nodes/move':>12s}  {'us/node':>8s}")
print("-" * 82)

# --- Baselines ---
run_benchmark("Random", 10000, play_game_random)

eng = Engine(0, 0, config)
run_benchmark("Heuristic (depth=0)", 1000, lambda: play_game(eng), eng)

# --- Constant-compute engine configs ---
NODES_PER_MOVE = 100000
print()
for depth in range(2, 6):
    fanout = compute_fanout(depth, NODES_PER_MOVE, config.num_bundles)
    eng = Engine(depth, fanout, config)
    run_benchmark(f"d={depth}, f={fanout:>4d}", 50, lambda e=eng: play_game(e), eng)
