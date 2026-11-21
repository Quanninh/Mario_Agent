import argparse
import json
import os
from pathlib import Path
import pickle
from typing import Optional, Type, Union
import pygame

from agent.tabular_q_agent import TabularQAgent
from agent.hybrid_q_sarsa_agent import HybridQSARSAAgent, HybridAgent
from core.base_env import Action
from core.interfaces import BaseAgent
from discretizer.state_discretizer import MarioStateDiscretizer
from env import create_env, get_level

ACTION_NAMES = {
    Action.STAY: "STAY",
    Action.LEFT: "LEFT",
    Action.JUMP: "JUMP",
    Action.RIGHT: "RIGHT",
    Action.RIGHT_JUMP: "RIGHT_JUMP",
}


def resolve_agent_class(
    agent_type: str = "auto",
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Type[BaseAgent]:
    """
    Determines whether to instantiate TabularQAgent or HybridAgent.
    Supports explicit CLI flags ('tabular'/'tabular_q', 'hybrid'/'hybrid_agent') and auto-detection.
    """
    clean_type = agent_type.strip().lower() if agent_type else "auto"

    if clean_type in ("hybrid", "hybridagent", "hybrid_agent", "hybrid_q_sarsa", "sarsa"):
        return HybridAgent

    if clean_type in ("tabular", "tabularqagent", "tabular_q", "q_learning", "q"):
        return TabularQAgent

    # Auto-detection mode: inspect checkpoint filename and serialized contents
    if checkpoint_path:
        path = Path(checkpoint_path)
        stem = path.stem.lower()

        # Filename hints
        if "hybrid" in stem or "sarsa" in stem:
            return HybridAgent

        # Serialized metadata inspection
        if path.exists():
            if path.suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and ("beta" in data or "mode" in data):
                        return HybridAgent
                except Exception:
                    pass
            elif path.suffix == ".pkl":
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and ("beta" in data or "mode" in data):
                        return HybridAgent
                except Exception:
                    pass

    return TabularQAgent


def resolve_checkpoint_path(
    mode: str,
    reward: str,
    variant: str,
    level: int = 1,
    explicit_path: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> Path:
    """Finds the checkpoint file path based on mode, reward, variant, level, and agent type with graceful fallbacks."""
    if explicit_path is not None:
        return Path(explicit_path)

    # If hybrid agent is explicitly requested, check for hybrid checkpoints first
    if agent_type and agent_type.lower() in ("hybrid", "hybridagent", "hybrid_agent", "sarsa"):
        p_hybrid = Path("checkpoints") / "hybrid_q_sarsa.pkl"
        if p_hybrid.exists():
            return p_hybrid
        p_hybrid_json = Path("checkpoints") / "hybrid_q_sarsa.json"
        if p_hybrid_json.exists():
            return p_hybrid_json

    if mode == "separate":
        # 1. Prefer checkpoints/separate_<reward>/q_table_lvl<N>_<variant>.json
        p_sep = Path("checkpoints") / f"separate_{reward}" / f"q_table_lvl{level}_{variant}.json"
        if p_sep.exists():
            return p_sep
        p_sep_pkl = Path("checkpoints") / f"separate_{reward}" / f"q_table_lvl{level}_{variant}.pkl"
        if p_sep_pkl.exists():
            return p_sep_pkl
        p_lvl = Path("checkpoints") / f"q_table_level_{level}_{variant}.json"
        if p_lvl.exists():
            return p_lvl
        p_lvl_pkl = Path("checkpoints") / f"q_table_level_{level}_{variant}.pkl"
        if p_lvl_pkl.exists():
            return p_lvl_pkl
        return p_sep

    # Curriculum mode:
    # 1. Prefer curriculum_<reward>/q_table_<variant>.json
    p1 = Path("checkpoints") / f"curriculum_{reward}" / f"q_table_{variant}.json"
    if p1.exists():
        return p1

    # 2. Fallback to root checkpoints/q_table_<variant>.json
    p2 = Path("checkpoints") / f"q_table_{variant}.json"
    if p2.exists():
        return p2

    # 3. Fallback to .pkl
    p3 = Path("checkpoints") / f"curriculum_{reward}" / f"q_table_{variant}.pkl"
    if p3.exists():
        return p3

    p4 = Path("checkpoints") / f"q_table_{variant}.pkl"
    if p4.exists():
        return p4

    p5 = Path("checkpoints") / f"q_table_level_{level}_{variant}.json"
    if p5.exists():
        return p5

    return p1


def load_agent_policy(
    agent: BaseAgent,
    mode: str,
    reward: str,
    variant: str,
    level: int = 1,
    explicit_path: Optional[str] = None,
) -> Path:
    """Loads checkpoint into agent policy, reporting state count and agent architecture."""
    cp_path = resolve_checkpoint_path(mode, reward, variant, level, explicit_path)
    if cp_path.exists():
        agent.load(str(cp_path))
        arch_name = "HybridAgent" if isinstance(agent, HybridAgent) else "TabularQAgent"
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
        choices=["auto", "tabular", "hybrid", "tabular_q", "hybrid_agent"],
        help="Agent architecture to use: 'tabular' (TabularQAgent), 'hybrid' (HybridAgent), or 'auto' (default: 'auto')",
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
        choices=[1, 2, 3, 4, 5],
        help="Initial level to load (1..5, default: 1)",
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
    return parser.parse_args()


def play_autonomous(
    checkpoint_path: Optional[str] = None,
    level_id: int = 1,
    agent_type: str = "auto",
    mode: str = "curriculum",
    reward: str = "balanced",
    variant: str = "best",
    fps: int = 60,
) -> None:
    """
    Main playback function for autonomous agents (TabularQAgent or HybridAgent).
    Can be called programmatically or via CLI through main().
    """
    current_mode = mode
    current_level = level_id
    current_reward = reward
    current_variant = variant
    current_agent_choice = agent_type
    explicit_checkpoint = checkpoint_path

    discretizer = MarioStateDiscretizer()

    # Determine initial checkpoint and agent class
    initial_cp = resolve_checkpoint_path(
        current_mode, current_reward, current_variant, current_level, explicit_checkpoint, current_agent_choice
    )
    agent_cls = resolve_agent_class(current_agent_choice, initial_cp)
    agent: BaseAgent = agent_cls()

    def reload_policy() -> Path:
        nonlocal agent
        target_cp = resolve_checkpoint_path(
            current_mode, current_reward, current_variant, current_level, explicit_checkpoint, current_agent_choice
        )
        target_cls = resolve_agent_class(current_agent_choice, target_cp)
        if not isinstance(agent, target_cls):
            agent = target_cls()
        return load_agent_policy(agent, current_mode, current_reward, current_variant, current_level, explicit_checkpoint)

    reload_policy()

    env = create_env(current_level)
    cfg = get_level(current_level)
    items_total = len(getattr(env, "items", []))

    agent_display_name = "HybridAgent (Hybrid Q-SARSA)" if isinstance(agent, HybridAgent) else "TabularQAgent (Pure Q-Learning)"

    print("\n" + "=" * 70)
    print(f"[AI Player] Playing Level {current_level}: {cfg.name}")
    print(f"[AI Player] Mode: {current_mode.upper()} | Reward: {current_reward.upper()} | Variant: {current_variant.upper()}")
    print(f"[AI Player] Active Agent: {agent_display_name}")
    print("Controls:")
    print("  [1 / 2 / 3 / 4 / 5] Switch Level live")
    print("  [F1 / F2 / F3]      Swap Q-Table: Balanced (F1), Sparse (F2), Dense (F3)")
    print("  [T]                 Toggle Variant: Best <-> Final")
    print("  [H / A]             Toggle Agent: TabularQAgent <-> HybridAgent")
    print("  [Space]             Toggle Pause / Resume")
    print("  [R]                 Reset Current Episode")
    print("  [Esc]               Quit")
    print("=" * 70 + "\n")

    raw_state = env.reset()
    discrete_state = discretizer.discretize(raw_state)
    steps = 0
    items_collected = 0
    paused = False
    action_name = "STAY"
    episode = 1

    # Initialize Pygame window
    env.render()
    prev_keys = list(pygame.key.get_pressed())
    clock = pygame.time.Clock() if fps != 60 else None

    while True:
        # Window closed inside env.render()
        if env._screen is None:
            break

        keys = pygame.key.get_pressed()
        just_pressed = lambda k: bool(keys[k] and not prev_keys[k])

        # Quit shortcut
        if just_pressed(pygame.K_ESCAPE):
            break

        # Level switching (1..5)
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

        if new_lvl is not None and new_lvl != current_level:
            env.close()
            current_level = new_lvl
            env = create_env(current_level)
            cfg = get_level(current_level)
            items_total = len(getattr(env, "items", []))
            print(f"\n[AI Player] Switched to Level {current_level}: {cfg.name}")
            if current_mode == "separate" and explicit_checkpoint is None:
                reload_policy()
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            prev_keys = list(pygame.key.get_pressed())
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
            discrete_state = discretizer.discretize(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            prev_keys = list(pygame.key.get_pressed())
            continue

        # Variant toggle (T)
        if just_pressed(pygame.K_t):
            current_variant = "final" if current_variant == "best" else "best"
            reload_policy()
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            prev_keys = list(pygame.key.get_pressed())
            continue

        # Agent architecture toggle (H or A)
        if just_pressed(pygame.K_h) or just_pressed(pygame.K_a):
            if isinstance(agent, HybridAgent):
                current_agent_choice = "tabular"
                agent = TabularQAgent()
            else:
                current_agent_choice = "hybrid"
                agent = HybridAgent()
            agent_display_name = "HybridAgent (Hybrid Q-SARSA)" if isinstance(agent, HybridAgent) else "TabularQAgent (Pure Q-Learning)"
            print(f"\n[AI Player] Switched Agent Architecture live: {agent_display_name}")
            reload_policy()
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)
            steps = 0
            items_collected = 0
            episode = 1
            env.render()
            prev_keys = list(pygame.key.get_pressed())
            continue

        # Pause toggle (Space)
        if just_pressed(pygame.K_SPACE):
            paused = not paused
            agent_tag = "HYBRID" if isinstance(agent, HybridAgent) else "TABULAR"
            caption = (
                f"Hoá's Midnight Escape | Lvl {current_level} | "
                f"{current_mode.upper()}-{current_reward.upper()} [{current_variant.upper()}] | "
                f"Agent: {agent_tag} | Step: {steps} | Items: {items_collected}/{items_total} | "
                f"Act: {action_name}{' | [PAUSED]' if paused else ''}"
            )
            pygame.display.set_caption(caption)

        # Reset episode (R)
        if just_pressed(pygame.K_r):
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)
            steps = 0
            items_collected = 0
            action_name = "STAY"
            print(f"[AI Player] Episode {episode} reset.")
            env.render()
            prev_keys = list(pygame.key.get_pressed())
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
        agent_tag = "HYBRID" if isinstance(agent, HybridAgent) else "TABULAR"
        caption = (
            f"Hoá's Midnight Escape | Lvl {current_level} | "
            f"{current_mode.upper()}-{current_reward.upper()} [{current_variant.upper()}] | "
            f"Agent: {agent_tag} | Step: {steps} | Items: {items_collected}/{items_total} | "
            f"Act: {action_name}"
        )
        pygame.display.set_caption(caption)

        next_raw_state, events, done = env.step(action)
        steps += 1
        if events.collected_item:
            items_collected += 1

        discrete_state = discretizer.discretize(next_raw_state)

        # Render frame (env.render() handles 60 FPS clock ticking and event pump)
        env.render()
        if clock is not None:
            clock.tick(fps)

        if done:
            if events.reached_goal:
                outcome_str = "WIN (Reached Home)"
            elif events.hit_ghost:
                outcome_str = "CAUGHT BY GHOST"
            elif events.fell_in_pit:
                outcome_str = "FELL IN PIT"
            else:
                outcome_str = "TIMEOUT"

            print(
                f"[AI Player] Episode {episode}: {outcome_str} in {steps} steps | "
                f"Items: {items_collected}/{items_total}"
            )
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)
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
        mode=args.mode,
        reward=args.reward,
        variant=args.variant,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
