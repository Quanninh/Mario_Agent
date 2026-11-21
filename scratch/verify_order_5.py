import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tabular_q_agent import TabularQAgent
from core.base_env import RawState
from discretizer.state_discretizer import MarioStateDiscretizer

def verify_all():
    print("--- 1. Testing Discretizer Overlap & Clearance ---")
    disc = MarioStateDiscretizer()
    s = RawState(
        player_x=100.0, player_y=150.0, velocity_y=5.0, is_grounded=False,
        dist_to_obstacle=-1.0, obstacle_height=0.0, dist_to_pit=-1.0,
        dist_to_ghost=0.0, ghost_y=200.0, dist_to_item=-1.0, dist_to_goal=500.0
    )
    d_state = disc.discretize(s)
    print("Discretized tuple:", d_state)
    assert len(d_state) == 7, f"Expected 7-tuple, got {len(d_state)}"
    assert d_state[5] == 3, f"Expected ghost_dist_bin == 3 (OVERLAP), got {d_state[5]}"
    print("[PASS] Check 1: ghost_dist_bin == 3 on dist_to_ghost == 0.0 with feet <= ghost_y.")

    print("\n--- 2. Testing outputs/training_board_curriculum_balanced.json ---")
    board_path = Path("outputs/training_board_curriculum_balanced.json")
    assert board_path.exists(), f"Board missing: {board_path}"
    with open(board_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    print("Metadata keys:", list(meta.keys()))
    assert meta["state_space_max"] == 1152, f"Expected state_space_max == 1152, got {meta.get('state_space_max')}"
    assert "alpha_start" in meta, "alpha_start missing from metadata"
    assert "alpha_min" in meta, "alpha_min missing from metadata"
    assert "rewarm_epsilon_ghost" in meta, "rewarm_epsilon_ghost missing from metadata"
    assert "replay_ratio" in meta, "replay_ratio missing from metadata"
    print(f"[PASS] Metadata fields verified: state_space_max={meta['state_space_max']}, alpha_start={meta['alpha_start']}, alpha_min={meta['alpha_min']}, rewarm_epsilon_ghost={meta['rewarm_epsilon_ghost']}, replay_ratio={meta['replay_ratio']}")

    history = data["episode_history"]
    print(f"Total episode_history rows: {len(history)}")
    assert len(history) == 25, f"Expected exactly 25 rows (5 eps/lvl), got {len(history)}"

    # Check level boundaries reset rolling_win_rate_50
    lvl2_ep1 = next(e for e in history if e["level_id"] == 2 and e["level_episode"] == 1)
    print("Level 2 Episode 1 rolling_win_rate_50:", lvl2_ep1["rolling_win_rate_50"])
    assert lvl2_ep1["rolling_win_rate_50"] == 0.0, f"Expected 0.0, got {lvl2_ep1['rolling_win_rate_50']}"

    # Check winning episodes have progress_pct == 100.0
    winning_eps = [e for e in history if e["won"]]
    print(f"Found {len(winning_eps)} winning episodes in smoke test.")
    for w in winning_eps:
        assert w["progress_pct"] == 100.0, f"Expected 100.0 progress on win, got {w['progress_pct']}"
    print("[PASS] Check 2: exactly 25 history rows, level resets verified, winning progress_pct == 100.0.")

    print("\n--- 3. Testing Root Checkpoint Load ---")
    for name in ["q_table_final.json", "q_table_final.pkl", "q_table_best.json", "q_table_best.pkl"]:
        cp_path = Path("checkpoints") / name
        assert cp_path.exists(), f"Missing checkpoint: {cp_path}"
        agent = TabularQAgent()
        agent.load(str(cp_path))
        print(f"Reloaded {name}: {len(agent.q_table)} states.")
        assert len(agent.q_table) > 0, f"Expected non-empty Q-table in {name}"
    print("[PASS] Check 3: Root checkpoints exist and load cleanly.")

    print("\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_all()
