import argparse
from pathlib import Path
from typing import Optional
import pygame

from agent.tabular_q_agent import TabularQAgent
from core.base_env import Action
from discretizer.state_discretizer import MarioStateDiscretizer
from env import create_env, get_level

ACTION_NAMES = {
    Action.STAY: "STAY",
    Action.LEFT: "LEFT",
    Action.JUMP: "JUMP",
    Action.RIGHT: "RIGHT",
    Action.RIGHT_JUMP: "RIGHT_JUMP",
}


def resolve_checkpoint_path(reward: str, variant: str, explicit_path: Optional[str] = None) -> Path:
    """Finds the checkpoint file path based on reward and variant with graceful fallbacks."""
    if explicit_path is not None:
        return Path(explicit_path)

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

    return p1


def load_agent_policy(agent: TabularQAgent, reward: str, variant: str, explicit_path: Optional[str] = None) -> Path:
    """Loads checkpoint into agent policy, reporting state count."""
    cp_path = resolve_checkpoint_path(reward, variant, explicit_path)
    if cp_path.exists():
        agent.load(str(cp_path))
        print(f"[AI Player] Loaded {reward.upper()} [{variant.upper()}] from {cp_path} ({len(agent.q_table)} states).")
    else:
        print(f"[AI Player] Note: Checkpoint '{cp_path}' not found. Policy is uninitialized.")
    agent.set_mode(training=False)
    return cp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="60 FPS Visual Playback Demo for Hoá's Midnight Escape")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    current_level = args.level
    current_reward = args.reward
    current_variant = args.variant

    discretizer = MarioStateDiscretizer()
    agent = TabularQAgent()

    load_agent_policy(agent, current_reward, current_variant, args.checkpoint)

    env = create_env(current_level)
    cfg = get_level(current_level)
    items_total = len(getattr(env, "items", []))

    print("\n" + "=" * 70)
    print(f"[AI Player] Playing Level {current_level}: {cfg.name}")
    print("Controls:")
    print("  [1 / 2 / 3 / 4 / 5] Switch Level live")
    print("  [F1 / F2 / F3]      Swap Q-Table: Balanced (F1), Sparse (F2), Dense (F3)")
    print("  [T]                 Toggle Variant: Best <-> Final")
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
            load_agent_policy(agent, current_reward, current_variant)
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
            load_agent_policy(agent, current_reward, current_variant)
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
            caption = (
                f"Hoá's Midnight Escape | Lvl {current_level} | "
                f"{current_reward.upper()} [{current_variant.upper()}] | "
                f"Step: {steps} | Items: {items_collected}/{items_total} | "
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
        caption = (
            f"Hoá's Midnight Escape | Lvl {current_level} | "
            f"{current_reward.upper()} [{current_variant.upper()}] | "
            f"Step: {steps} | Items: {items_collected}/{items_total} | "
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


if __name__ == "__main__":
    main()
