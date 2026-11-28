import argparse
import json
import os
from pathlib import Path
import pickle
import random
from typing import List, Optional, Type, Union
import pygame

from agent.tabular_q_agent import TabularQAgent
from agent.hybrid_q_sarsa_agent import HybridQSARSAAgent, HybridAgent
from agent.dqn_agent import DQNAgent
from core.base_env import Action
from core.interfaces import BaseAgent
from discretizer.state_discretizer import MarioStateDiscretizer
from discretizer.vector_extractor import ContinuousStateExtractor
from env import create_env, get_level

ACTION_NAMES = {
    Action.STAY: "STAY",
    Action.LEFT: "LEFT",
    Action.JUMP: "JUMP",
    Action.RIGHT: "RIGHT",
    Action.RIGHT_JUMP: "RIGHT_JUMP",
    Action.CROUCH: "CROUCH",
}


def resolve_agent_class(
    agent_type: str = "auto",
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Type[BaseAgent]:
    """
    Determines whether to instantiate TabularQAgent or HybridQSARSAAgent.
    Supports explicit CLI flags ('tabular'/'tabular_q', 'hybrid'/'hybrid_agent') and auto-detection.
    """
    clean_type = agent_type.strip().lower() if agent_type else "auto"

    if clean_type in ("dqn", "dqn_agent", "deep_q", "d3qn"):
        return DQNAgent

    if clean_type in ("hybrid", "hybridagent", "hybrid_agent", "hybrid_q_sarsa", "sarsa"):
        return HybridQSARSAAgent

    if clean_type in ("tabular", "tabularqagent", "tabular_q", "q_learning", "q"):
        return TabularQAgent

    # Auto-detection mode: inspect checkpoint filename and serialized contents
    if checkpoint_path:
        path = Path(checkpoint_path)
        parts = [p.lower() for p in path.parts]

        # PyTorch checkpoint detection
        if path.suffix == ".pt" or any("dqn" in p for p in parts):
            return DQNAgent

        # Directory / Filename hints
        if any("hybrid" in p or "sarsa" in p for p in parts):
            return HybridQSARSAAgent

        # Serialized metadata inspection
        if path.exists():
            if path.suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and ("beta" in data or "mode" in data or ("metadata" in data and "hybrid" in str(data["metadata"]))):
                        return HybridQSARSAAgent
                except Exception:
                    pass
            elif path.suffix == ".pkl":
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and ("beta" in data or "mode" in data or "agent_type" in data):
                        return HybridQSARSAAgent
                except Exception:
                    pass

    return TabularQAgent


def make_agent_instance(agent_cls: Type[BaseAgent], hybrid_mode: str = "expected") -> BaseAgent:
    """Helper to instantiate agent class with hybrid mode if applicable."""
    if issubclass(agent_cls, DQNAgent):
        return agent_cls()
    if issubclass(agent_cls, HybridQSARSAAgent):
        return agent_cls(mode=hybrid_mode)
    return agent_cls()


def resolve_checkpoint_path(
    mode: str,
    reward: str,
    variant: str,
    level: int = 1,
    explicit_path: Optional[str] = None,
    agent_type: Optional[str] = None,
    hybrid_mode: str = "expected",
) -> Path:
    """Finds the checkpoint file path based on mode, reward, variant, level, agent type, and hybrid mode with graceful fallbacks."""
    if explicit_path is not None:
        return Path(explicit_path)

    clean_agent = (agent_type or "auto").strip().lower()
    if clean_agent in ("hybrid", "hybridagent", "hybrid_agent", "sarsa"):
        agent_tags = [f"hybrid_{hybrid_mode}", "hybrid"]
    elif clean_agent in ("tabular", "tabularqagent", "tabular_q", "q"):
        agent_tags = ["tabular"]
    else:
        # Auto: check hybrid tag first, then tabular
        agent_tags = [f"hybrid_{hybrid_mode}", "tabular", "hybrid"]

    # 1. Search in checkpoints/<agent_tag>/<mode>_<reward>/
    for tag in agent_tags:
        tagged_dir = Path("checkpoints") / tag / f"{mode}_{reward}"
        if mode == "separate":
            candidates = [
                tagged_dir / f"q_table_lvl{level}_{variant}.json",
                tagged_dir / f"q_table_lvl{level}_{variant}.pkl",
                tagged_dir / f"q_table_lvl{level}_best.json" if variant == "best" else None,
                tagged_dir / f"q_table_lvl{level}_best.pkl" if variant == "best" else None,
            ]
        else:
            candidates = [
                tagged_dir / f"q_table_lvl{level}_{variant}.json",
                tagged_dir / f"q_table_lvl{level}_{variant}.pkl",
                tagged_dir / f"q_table_{variant}.json",
                tagged_dir / f"q_table_{variant}.pkl",
            ]
        for c in candidates:
            if c is not None and c.exists():
                return c

    # 2. Search in legacy non-tagged directory: checkpoints/<mode>_<reward>/
    legacy_dir = Path("checkpoints") / f"{mode}_{reward}"
    if mode == "separate":
        legacy_candidates = [
            legacy_dir / f"q_table_lvl{level}_{variant}.json",
            legacy_dir / f"q_table_lvl{level}_{variant}.pkl",
        ]
    else:
        legacy_candidates = [
            legacy_dir / f"q_table_lvl{level}_{variant}.json",
            legacy_dir / f"q_table_lvl{level}_{variant}.pkl",
            legacy_dir / f"q_table_{variant}.json",
            legacy_dir / f"q_table_{variant}.pkl",
        ]
    for c in legacy_candidates:
        if c.exists():
            return c

    # 3. Search in root checkpoints/
    root_candidates = [
        Path("checkpoints") / f"q_table_{variant}.json",
        Path("checkpoints") / f"q_table_{variant}.pkl",
        Path("checkpoints") / f"q_table_level_{level}_{variant}.json",
        Path("checkpoints") / f"q_table_level_{level}_{variant}.pkl",
        Path("checkpoints") / "hybrid_q_sarsa.pkl",
        Path("checkpoints") / "hybrid_q_sarsa.json",
    ]
    for c in root_candidates:
        if c.exists():
            return c

    # 4. Default expected path when nothing exists yet
    primary_tag = agent_tags[0]
    primary_dir = Path("checkpoints") / primary_tag / f"{mode}_{reward}"
    if mode == "separate":
        return primary_dir / f"q_table_lvl{level}_{variant}.json"
    return primary_dir / f"q_table_{variant}.json"


def load_agent_policy(
    agent: BaseAgent,
    mode: str,
    reward: str,
    variant: str,
    level: int = 1,
    explicit_path: Optional[str] = None,
    agent_type: Optional[str] = None,
    hybrid_mode: str = "expected",
) -> Path:
    """Loads checkpoint into agent policy, reporting state count and agent architecture with uninitialized warnings."""
    cp_path = resolve_checkpoint_path(
        mode, reward, variant, level, explicit_path, agent_type, hybrid_mode
    )
    if cp_path.exists():
        agent.load(str(cp_path))
        if isinstance(agent, DQNAgent):
            arch_name = "DQNAgent (Dueling Double DQN)"
            print(f"[AI Player] Loaded neural policy ({arch_name}) from {cp_path}.")
        else:
            arch_name = "HybridAgent" if isinstance(agent, (HybridAgent, HybridQSARSAAgent)) else "TabularQAgent"
            if len(getattr(agent, "q_table", {})) == 0:
                print(
                    f"[AI Player] Warning: Loaded '{cp_path}' but Q-table has 0 states (empty/uninitialized policy)."
                )
            else:
                print(
                    f"[AI Player] Loaded {mode.upper()} {reward.upper()} [{variant.upper()}] "
                    f"({arch_name}) from {cp_path} ({len(agent.q_table)} states)."
                )
    else:
        print(f"[AI Player] Note: Checkpoint '{cp_path}' not found. Policy is uninitialized.")
    agent.set_mode(training=False)
    return cp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="60 FPS Visual Playback Demo for Hoá's Midnight Escape")
    parser.add_argument(
        "--agent",
        type=str,
        default="auto",
        choices=["auto", "tabular", "hybrid", "tabular_q", "hybrid_agent", "dqn", "dqn_agent"],
        help="Agent architecture to use: 'tabular', 'hybrid', 'dqn', or 'auto' (default: 'auto')",
    )
    parser.add_argument(
        "--hybrid-mode",
        type=str,
        default="expected",
        choices=["expected", "linear", "hazard_gated"],
        help="Target calculation mode for Hybrid agent: 'expected', 'linear', or 'hazard_gated' (default: 'expected')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="curriculum",
        choices=["curriculum", "separate"],
        help="Training mode checkpoint to load: 'curriculum' or 'separate' (default: 'curriculum')",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Initial level to load (1..8, default: 1)",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="balanced",
        choices=["balanced", "sparse", "dense"],
        help="Reward strategy checkpoint to load (default: 'balanced')",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Model variant to load: 'best' or 'final' (default: 'best')",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path override to a .json or .pkl checkpoint file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render frame rate (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional procedural LevelGenerator seed (Levels 1-5)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of procedural seeds to cycle through after each episode (e.g. --seeds 51 55 59)",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Automatically pick a new random procedural seed after each episode (Levels 1-5)",
    )
    return parser.parse_args()


def play_autonomous(
    checkpoint_path: Optional[str] = None,
    level_id: int = 1,
    agent_type: str = "auto",
    hybrid_mode: str = "expected",
    mode: str = "curriculum",
    reward: str = "balanced",
    variant: str = "best",
    fps: int = 60,
    seed: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    random_seed: bool = False,
) -> None:
    """
    Main playback function for autonomous agents (TabularQAgent, HybridAgent, or DQNAgent).
    Can be called programmatically or via CLI through main().
    """
    current_mode = mode
    current_level = level_id
    current_reward = reward
    current_variant = variant
    current_agent_choice = agent_type
    current_hybrid_mode = hybrid_mode
    seed_list = list(seeds) if seeds else None
    seed_idx = 0
    if seed_list:
        current_seed: Optional[int] = seed_list[0]
    elif random_seed and seed is None and 1 <= current_level <= 5:
        current_seed = random.randint(1, 999999)
    else:
        current_seed = seed
    explicit_checkpoint = checkpoint_path

    discretizer = MarioStateDiscretizer()
    vector_extractor = ContinuousStateExtractor()

    def get_agent_state(rs):
        if isinstance(agent, DQNAgent):
            return vector_extractor.extract(rs)
        return discretizer.discretize(rs)

    # Determine initial checkpoint and agent class
    initial_cp = resolve_checkpoint_path(
        current_mode, current_reward, current_variant, current_level, explicit_checkpoint, current_agent_choice, current_hybrid_mode
    )
    agent_cls = resolve_agent_class(current_agent_choice, initial_cp)
    agent: BaseAgent = make_agent_instance(agent_cls, current_hybrid_mode)

    def reload_policy() -> Path:
        nonlocal agent
        target_cp = resolve_checkpoint_path(
            current_mode, current_reward, current_variant, current_level, explicit_checkpoint, current_agent_choice, current_hybrid_mode
        )
        target_cls = resolve_agent_class(current_agent_choice, target_cp)
        if not isinstance(agent, target_cls):
            agent = make_agent_instance(target_cls, current_hybrid_mode)
        return load_agent_policy(
            agent, current_mode, current_reward, current_variant, current_level, explicit_checkpoint, current_agent_choice, current_hybrid_mode
        )

    reload_policy()

    env = create_env(current_level, seed=current_seed)
    cfg = get_level(current_level, seed=current_seed)
    items_total = len(getattr(env, "items", []))

    if isinstance(agent, DQNAgent):
        agent_display_name = "DQNAgent (Dueling Double DQN)"
    elif isinstance(agent, (HybridAgent, HybridQSARSAAgent)):
        agent_display_name = f"HybridAgent (Hybrid Q-SARSA: {current_hybrid_mode})"
    else:
        agent_display_name = "TabularQAgent (Pure Q-Learning)"

    print("\n" + "=" * 70)
    print(f"[AI Player] Playing Level {current_level}: {cfg.name} (seed={current_seed})")
    print(f"[AI Player] Mode: {current_mode.upper()} | Reward: {current_reward.upper()} | Variant: {current_variant.upper()}")
    print(f"[AI Player] Active Agent: {agent_display_name}")
    print("Controls:")
    print("  [1 - 8]             Switch Level live (1..8)")
    print("  [N / C]             New random procedural seed (N) / Canonical static map (C)")
    print("  [F1 / F2 / F3]      Swap Q-Table: Balanced (F1), Sparse (F2), Dense (F3)")
    print("  [T]                 Toggle Variant: Best <-> Final")
    print("  [H / A]             Toggle Agent: TabularQAgent <-> HybridAgent")
    print("  [Space]             Toggle Pause / Resume")
    print("  [R]                 Reset Current Episode")
    print("  [Esc]               Quit")
    print("=" * 70 + "\n")

    raw_state = env.reset()
    discrete_state = get_agent_state(raw_state)
    steps = 0
    items_collected = 0
    paused = False
    action_name = "STAY"
    episode = 1

    # Initialize Pygame window
    env.render()
    if not getattr(env, "is_running", True):
        return
    try:
        prev_keys = list(pygame.key.get_pressed())
    except pygame.error:
        return
    clock = pygame.time.Clock() if fps != 60 else None

    while True:
        # Window closed inside env.render()
        if not getattr(env, "is_running", True):
            break

        try:
            keys = pygame.key.get_pressed()
        except pygame.error:
            break
        just_pressed = lambda k: bool(keys[k] and (k < len(prev_keys) and not prev_keys[k]))

        # Quit shortcut
        if just_pressed(pygame.K_ESCAPE):
            break

        # Level switching (1..8)
        new_lvl = None
        if just_pressed(pygame.K_1):
            new_lvl = 1
        elif just_pressed(pygame.K_2):
            new_lvl = 2
        elif just_pressed(pygame.K_3):
            new_lvl = 3
        elif just_pressed(pygame.K_4):
            new_lvl = 4
        elif just_pressed(pygame.K_5):
            new_lvl = 5
        elif just_pressed(pygame.K_6):
            new_lvl = 6
        elif just_pressed(pygame.K_7):
            new_lvl = 7
        elif just_pressed(pygame.K_8):
            new_lvl = 8

        if new_lvl is not None and new_lvl != current_level:
            env.close()
            current_level = new_lvl
            env = create_env(current_level, seed=current_seed)
            cfg = get_level(current_level, seed=current_seed)
            items_total = len(getattr(env, "items", []))
            print(f"\n[AI Player] Switched to Level {current_level}: {cfg.name} (seed={current_seed})")
            if explicit_checkpoint is None:
                reload_policy()
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        # Procedural seed controls (N = random seed, C = canonical static map)
        if just_pressed(pygame.K_n) or just_pressed(pygame.K_c):
            current_seed = random.randint(1, 999999) if just_pressed(pygame.K_n) else None
            env.close()
            env = create_env(current_level, seed=current_seed)
            cfg = get_level(current_level, seed=current_seed)
            items_total = len(getattr(env, "items", []))
            print(f"\n[AI Player] Reloaded Level {current_level}: {cfg.name} (seed={current_seed})")
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        # Strategy swapping (F1..F3)
        new_reward = None
        if just_pressed(pygame.K_F1):
            new_reward = "balanced"
        elif just_pressed(pygame.K_F2):
            new_reward = "sparse"
        elif just_pressed(pygame.K_F3):
            new_reward = "dense"

        if new_reward is not None and new_reward != current_reward:
            current_reward = new_reward
            reload_policy()
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        # Variant toggle (T)
        if just_pressed(pygame.K_t):
            current_variant = "final" if current_variant == "best" else "best"
            reload_policy()
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        # Agent architecture toggle (H or A)
        if just_pressed(pygame.K_h) or just_pressed(pygame.K_a):
            if isinstance(agent, (HybridAgent, HybridQSARSAAgent)):
                current_agent_choice = "tabular"
                agent = TabularQAgent()
            else:
                current_agent_choice = "hybrid"
                agent = HybridQSARSAAgent(mode=current_hybrid_mode)
            agent_display_name = (
                f"HybridAgent (Hybrid Q-SARSA: {current_hybrid_mode})"
                if isinstance(agent, (HybridAgent, HybridQSARSAAgent))
                else "TabularQAgent (Pure Q-Learning)"
            )
            print(f"\n[AI Player] Switched Agent Architecture live: {agent_display_name}")
            reload_policy()
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        # Pause toggle (Space)
        if just_pressed(pygame.K_SPACE):
            paused = not paused
            agent_tag_str = (
                f"HYBRID ({current_hybrid_mode})"
                if isinstance(agent, (HybridAgent, HybridQSARSAAgent))
                else "TABULAR"
            )
            caption = (
                f"Hoá's Midnight Escape | Lvl {current_level} | "
                f"{current_mode.upper()}-{current_reward.upper()} [{current_variant.upper()}] | "
                f"Agent: {agent_tag_str} | Step: {steps} | Items: {items_collected}/{items_total} | "
                f"Act: {action_name}{' | [PAUSED]' if paused else ''}"
            )
            pygame.display.set_caption(caption)

        # Reset episode (R)
        if just_pressed(pygame.K_r):
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            action_name = "STAY"
            print(f"[AI Player] Episode {episode} reset.")
            env.render()
            if not getattr(env, "is_running", True):
                break
            prev_keys = list(keys)
            continue

        prev_keys = list(keys)

        # While paused, do not step physics or call env.render()
        if paused:
            for ev in pygame.event.get(pygame.QUIT):
                if ev.type == pygame.QUIT:
                    env.close()
                    return
            pygame.event.pump()
            pygame.time.wait(30)
            continue

        # ------------------------------------
        # Active Playback: Step Agent & Physics
        # ------------------------------------
        action = agent.select_action(discrete_state)
        action_name = ACTION_NAMES.get(action, str(action))

        # Update caption with telemetry
        agent_tag_str = (
            f"HYBRID ({current_hybrid_mode})"
            if isinstance(agent, (HybridAgent, HybridQSARSAAgent))
            else "TABULAR"
        )
        caption = (
            f"Hoá's Midnight Escape | Lvl {current_level} | "
            f"{current_mode.upper()}-{current_reward.upper()} [{current_variant.upper()}] | "
            f"Agent: {agent_tag_str} | Step: {steps} | Items: {items_collected}/{items_total} | "
            f"Act: {action_name}"
        )
        pygame.display.set_caption(caption)

        next_raw_state, events, done = env.step(action)
        steps += 1
        if events.collected_item:
            items_collected += 1

        discrete_state = get_agent_state(next_raw_state)

        # Render frame (env.render() handles 60 FPS clock ticking and event pump)
        env.render()
        if not getattr(env, "is_running", True):
            break
        if clock is not None:
            clock.tick(fps)

        if done:
            if events.reached_goal:
                outcome_str = "WIN (Reached Home)"
            elif events.hit_ghost:
                outcome_str = "CAUGHT BY GHOST"
            elif events.fell_in_pit:
                outcome_str = "FELL IN PIT"
            elif getattr(events, "hit_fire", False):
                outcome_str = "INCINERATED BY PIT FIRE"
            elif getattr(events, "hit_laser", False):
                outcome_str = "VAPORIZED BY LASER"
            elif getattr(events, "hit_wisp", False):
                outcome_str = "STRUCK BY WILL-O'-THE-WISP"
            elif getattr(events, "hit_monster", False):
                monster_type = getattr(events, "monster_type", None)
                if monster_type == "cuong_thi":
                    outcome_str = "AMBUSHED BY CƯƠNG THI"
                elif monster_type == "quy_nhap_trang":
                    outcome_str = "SLAIN BY QUỶ NHẬP TRÀNG"
                else:
                    outcome_str = "AMBUSHED BY GRAVE MONSTER"
            else:
                outcome_str = "TIMEOUT"

            print(
                f"[AI Player] Episode {episode}: {outcome_str} in {steps} steps | "
                f"Items: {items_collected}/{items_total} | Seed: {current_seed}"
            )
            if (seed_list or random_seed) and 1 <= current_level <= 5:
                if seed_list:
                    seed_idx = (seed_idx + 1) % len(seed_list)
                    next_seed = seed_list[seed_idx]
                else:
                    next_seed = random.randint(1, 999999)
                if next_seed != current_seed:
                    current_seed = next_seed
                    env.close()
                    env = create_env(current_level, seed=current_seed)
                    cfg = get_level(current_level, seed=current_seed)
                    items_total = len(getattr(env, "items", []))
                    print(f"[AI Player] Next Episode Seed -> {current_seed}: {cfg.name}")
            raw_state = env.reset()
            discrete_state = get_agent_state(raw_state)
            steps = 0
            items_collected = 0
            episode += 1
            pygame.time.wait(500)

    env.close()
    print("[AI Player] Window closed. Goodbye!")


def main() -> None:
    args = parse_args()
    play_autonomous(
        checkpoint_path=args.checkpoint,
        level_id=args.level,
        agent_type=args.agent,
        hybrid_mode=args.hybrid_mode,
        mode=args.mode,
        reward=args.reward,
        variant=args.variant,
        fps=args.fps,
        seed=args.seed,
        seeds=args.seeds,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
