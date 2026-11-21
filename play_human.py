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
        f"Ghosts: {len(cfg.ghosts)}, Warmth Items: {len(cfg.items)})"
    )
    return env, state

def show_overlay(env, title: str, subtitle: str, color=(255, 255, 255), duration_ms: int = 1400) -> None:
    """Render a semi-transparent dark banner overlay and pause briefly before reset."""
    if env._screen is None:
        return

    sw, sh = env.screen_width, env.screen_height

    # Semi-transparent dark overlay over the frozen frame
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    env._screen.blit(overlay, (0, 0))

    pygame.font.init()
    title_font = pygame.font.SysFont("Arial", 44, bold=True)
    sub_font = pygame.font.SysFont("Arial", 18, bold=True)

    # Title with drop shadow
    t_surf = title_font.render(title, True, color)
    t_rect = t_surf.get_rect(center=(sw // 2, sh // 2 - 22))
    t_shadow = title_font.render(title, True, (10, 10, 10))
    env._screen.blit(t_shadow, (t_rect.x + 2, t_rect.y + 2))
    env._screen.blit(t_surf, t_rect)

    # Subtitle with drop shadow
    s_surf = sub_font.render(subtitle, True, (230, 235, 245))
    s_rect = s_surf.get_rect(center=(sw // 2, sh // 2 + 26))
    s_shadow = sub_font.render(subtitle, True, (10, 10, 10))
    env._screen.blit(s_shadow, (s_rect.x + 1, s_rect.y + 1))
    env._screen.blit(s_surf, s_rect)

    pygame.display.flip()

    # Event-pumping pause so window never freezes during the overlay
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < duration_ms:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
                elif event.key in (pygame.K_r, pygame.K_SPACE, pygame.K_RETURN):
                    return
        pygame.time.delay(20)


def main() -> None:
    current_level = 1
    env, state = load_level(current_level)

    print("Controls:")
    print("  [A / Left]       Walk Left")
    print("  [D / Right]      Walk Right")
    print("  [Space / W]      Jump")
    print("  [1 / 2 / 3 / 4 / 5] Switch Level 1 to 5")
    print("  [R]              Reset Level")
    print("  [Esc]            Quit")

    level_keys = {
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
    }

    jump_key_pressed = False

    while True:
        # Stop cleanly if window was closed inside env.render()
        if env._screen is None:
            break

        keys = pygame.key.get_pressed()

        # Quit shortcut
        if keys[pygame.K_ESCAPE]:
            break

        # Level switching shortcuts (1 to 5)
        selected_level = next((lvl for k, lvl in level_keys.items() if keys[k]), None)
        if selected_level is not None and selected_level != current_level:
            env.close()
            current_level = selected_level
            env, state = load_level(current_level)
            continue

        # Manual reset shortcut (render() pumps Pygame events so window won't freeze)
        if keys[pygame.K_r]:
            state = env.reset()
            env.render()
            continue

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
            if events.reached_goal:
                collected = sum(1 for item in env.items if item.is_collected)
                total = len(env.items)
                warmth_info = f" • Warmth: {collected}/{total}" if total > 0 else ""
                show_overlay(
                    env,
                    title="HOME SAFE!",
                    subtitle=f"Cleared {env.level_name} in {env.step_count} steps!{warmth_info}",
                    color=(110, 245, 150),  # Radiant emerald green
                    duration_ms=5000,
                )
            elif events.hit_fire:
                show_overlay(
                    env,
                    title="INCINERATED!",
                    subtitle="Burned by the grave's inferno!",
                    color=(255, 95, 30),  # Intense fiery orange
                    duration_ms=3000,
                )
            elif events.hit_ghost:
                show_overlay(
                    env,
                    title="GAME OVER",
                    subtitle="Caught by a spectral ghost!",
                    color=(255, 80, 80),   # Crimson red
                    duration_ms=3000,
                )
            elif events.fell_in_pit:
                show_overlay(
                    env,
                    title="GAME OVER",
                    subtitle="Fell into an open grave pit!",
                    color=(255, 80, 80),
                    duration_ms=3000,
                )
            else:
                show_overlay(
                    env,
                    title="TIME OUT",
                    subtitle="The freezing midnight overcame Hoá...",
                    color=(255, 185, 70),  # Cold amber
                    duration_ms=3000,
                )
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()
