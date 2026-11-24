import pygame
from core import Action
from env import create_env, get_level


def load_level(level_id: int):
    """Instantiate, reset, and open the Pygame window for the given level."""
    cfg = get_level(level_id)
    env = create_env(level_id)
    state = env.reset()
    env.render()
    print(
        f"\nLoaded {cfg.name} "
        f"(Tombstones: {len(cfg.obstacles)}, Pits: {len(cfg.pits)}, "
        f"Ghosts: {len(cfg.ghosts)}, Monsters: {len(cfg.monsters or [])}, "
        f"Warmth Items: {len(cfg.items)})"
    )
    return env, state


def main() -> None:
    current_level = 1
    env, state = load_level(current_level)

    print("Controls:")
    print("  [A / Left]       Walk Left")
    print("  [D / Right]      Walk Right")
    print("  [Space / W]      Jump")
    print("  [1 / 2 / 3 / 4 / 5 / 6 / 7] Switch Level 1 to 7")
    print("  [G]              Toggle God Mode")
    print("  [R]              Reset Level")
    print("  [Esc]            Quit")

    level_keys = {
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
        pygame.K_7: 7,
        pygame.K_8: 8,
    }

    jump_key_pressed = False
    god_key_pressed = False

    while True:
        # Stop cleanly if window was closed inside env.render()
        if not env.is_running:
            break

        keys = pygame.key.get_pressed()

        # Quit shortcut
        if keys[pygame.K_ESCAPE]:
            break

        # Level switching shortcuts (1 to 6)
        selected_level = next((lvl for k, lvl in level_keys.items() if keys[k]), None)
        if selected_level is not None and selected_level != current_level:
            was_god_mode = getattr(env, "god_mode", False)
            env.close()
            current_level = selected_level
            env, state = load_level(current_level)
            env.god_mode = was_god_mode
            continue

        # Manual reset shortcut (render() pumps Pygame events so window won't freeze)
        if keys[pygame.K_r]:
            state = env.reset()
            env.render()
            continue

        # Toggle God Mode shortcut [G]
        god_key_down = keys[pygame.K_g]
        if god_key_down and not god_key_pressed:
            env.god_mode = not env.god_mode
            status = "ENABLED (Invincible)" if env.god_mode else "DISABLED (Normal)"
            print(f"\n>>> GOD MODE: {status} <<<")
        god_key_pressed = god_key_down

        

        # Map keyboard input -> Discrete Action ID
        moving_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        moving_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        jump_key_down = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
        crouching = keys[pygame.K_DOWN] or keys[pygame.K_s]

        # Grounded: holding jump triggers initial jump.
        # Airborne: requires a fresh tap to trigger mid-air double jump.
        if env.player.is_grounded:
            jumping = jump_key_down
        else:
            jumping = jump_key_down and not jump_key_pressed

        jump_key_pressed = jump_key_down

        if crouching and env.player.is_grounded:
            action = Action.CROUCH
        elif moving_right and jumping:
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

        if events.collected_item:
            print(f"Warmth item collected at x={state.player_x:.1f}!")

        if done:
            env.show_game_over(events)
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()
