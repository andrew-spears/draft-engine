from game import GameConfig, play_game
from engine import Engine

config = GameConfig()

DEPTH = 3
FANOUT = 10

eng = Engine(DEPTH, FANOUT, config)

print(f"Full game playthrough: depth={DEPTH}, fanout={FANOUT}")
print(f"Config: draw={config.draw_size}, bundles={config.num_bundles}, overlap={config.overlap_degree}")
print()

play_game(config, eng.get_action, verbose=True)
print(f"  Nodes searched: {eng.node_count}, avg {eng.node_count / config.num_rounds:.1f} per move")
