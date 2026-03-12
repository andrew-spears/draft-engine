import numpy as np
import time
from game import GameConfig, play_game
from engine import Engine

config = GameConfig()


def run_benchmark(name, n, action_fn, eng=None):
    start = time.time()
    scores = [play_game(config, action_fn) for _ in range(n)]
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
play_game(config, Engine(1, 5, config).get_action)
print("done.\n")

# --- Header ---
print(f"{'Strategy':>25s}  {'n':>5s}  {'mean':>6s}  {'std':>5s}  {'s/game':>8s}  {'nodes/move':>12s}  {'us/node':>8s}")
print("-" * 82)

# --- Baselines ---
run_benchmark("Random", 10000, lambda t: np.random.randint(len(t)))

eng = Engine(0, 0, config)
run_benchmark("Heuristic (depth=0)", 1000, eng.get_action, eng)

# --- Constant-compute engine configs ---
NODES_PER_MOVE = 100000
print()
for depth in range(2, 6):
    fanout = compute_fanout(depth, NODES_PER_MOVE, config.num_bundles)
    eng = Engine(depth, fanout, config)
    run_benchmark(f"d={depth}, f={fanout:>4d}", 50, eng.get_action, eng)
