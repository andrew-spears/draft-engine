import numpy as np
from game import GameConfig, sample_transitions, total_score_from_table
from engine import Engine

config = GameConfig()
score_table = config.make_score_table()

DEPTH = 3
FANOUT = 10


def format_bundles(draw, assignments, action, config):
    """Show each bundle as a row, with the chosen one marked."""
    lines = []
    for b in range(config.num_bundles):
        goods = []
        for j in range(config.draw_size):
            for r in range(config.overlap_degree):
                if assignments[j, r] == b:
                    goods.append(str(draw[j] + 1))
                    break
        contents = " ".join(goods) if goods else "(empty)"
        marker = " >>>" if b == action else ""
        lines.append(f"           bundle {b}: {contents}{marker}")
    return "\n".join(lines)


eng = Engine(DEPTH, FANOUT, config)

print(f"Full game playthrough: depth={DEPTH}, fanout={FANOUT}")
print(f"Config: draw={config.draw_size}, bundles={config.num_bundles}, overlap={config.overlap_degree}")
print()

stashed = config.init_stashed
remaining = config.init_pool

for round_num in range(config.num_rounds):
    transitions, draw, assignments = sample_transitions(stashed, remaining, config)
    action = eng.get_action(transitions)
    stashed, remaining = transitions[action]

    draw_str = " ".join(str(v + 1) for v in draw)
    stash_str = "  ".join(f"{c:>2}" for c in stashed)
    remaining_str = "  ".join(f"{c:>2}" for c in remaining)
    type_header = "  ".join(f"{i+1:>2}" for i in range(config.num_types))

    print(f"  Round {round_num+1}: drew [{draw_str}]")
    print(format_bundles(draw, assignments, action, config))
    print(f"  Stash:     [{stash_str}]")
    print(f"  Remaining: [{remaining_str}]")
    print()

final_score = total_score_from_table(np.array(stashed, dtype=np.int64), score_table)
print(f"  Score: {final_score:.0f}")
print(f"  Nodes searched: {eng.node_count}, avg {eng.node_count / config.num_rounds:.1f} per move")
