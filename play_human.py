import pygame
from core import Action
from env import create_env, get_level


def load_level(level_id: int):
    """Instantiate, reset, and open the Pygame window for the given level."""
    cfg = get_level(level_id)
    env = create_env(level_id)
    state = env.reset()
    env.render()
    print(f"\nLoaded {cfg.name} (Obstacles: {len(cfg.obstacles)}, Pits: {len(cfg.pits)})")
    return env, state


def main() -> None:
    current_level = 1
    env, state = load_level(current_level)

    print("Controls:")
    print("  [A / Left]   Walk Left")
    print("  [D / Right]  Walk Right")
    print("  [Space / W]  Jump")
    print("  [1 / 2 / 3]  Switch Level 1, 2, or 3")
    print("  [R]          Reset Level")
    print("  [Esc]        Quit")

    prev_r_pressed = False

    while True:
        # Stop cleanly if window was closed inside env.render()
        if env._screen is None:
            break

        keys = pygame.key.get_pressed()

        # Quit shortcut
        if keys[pygame.K_ESCAPE]:
            break

        # Level switching shortcuts (1, 2, 3)
        selected_level = None
        if keys[pygame.K_1]:
            selected_level = 1
        elif keys[pygame.K_2]:
            selected_level = 2
        elif keys[pygame.K_3]:
            selected_level = 3

        if selected_level is not None and selected_level != current_level:
            env.close()
            current_level = selected_level
            env, state = load_level(current_level)
            prev_r_pressed = False
            continue

        # Manual reset shortcut (rising-edge trigger + render to pump Pygame events)
        r_pressed = keys[pygame.K_r]
        if r_pressed and not prev_r_pressed:
            state = env.reset()
            env.render()
            prev_r_pressed = r_pressed
            continue
        prev_r_pressed = r_pressed

        # Map keyboard input -> Discrete Action ID
        moving_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        moving_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        jumping = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]

        if moving_right and jumping:
            action = Action.RIGHT_JUMP
        elif moving_right:
            action = Action.RIGHT
        elif moving_left:
            action = Action.LEFT
        elif jumping:
            action = Action.JUMP
        else:
            action = Action.STAY

        # Step physics & render frame
        state, events, done = env.step(action)
        env.render()

        if done:
            if events.reached_goal:
                print(f"Cleared Level {current_level} in {env.step_count} steps!")
            elif events.fell_in_pit:
                print(f"Fell into a pit at x={state.mario_x:.1f}!")
            else:
                print("Timeout!")
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()
