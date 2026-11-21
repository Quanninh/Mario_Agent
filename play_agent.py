import os
import time
from pathlib import Path
import pygame

from agent.tabular_q_agent import TabularQAgent
from discretizer.state_discretizer import MarioStateDiscretizer
from env import create_env, get_level


def play_autonomous(checkpoint_path: str = "checkpoints/q_table_best.json", level_id: int = 1):
    # 1. Instantiate Environment, Discretizer, and Agent
    cfg = get_level(level_id)
    env = create_env(level_id)
    discretizer = MarioStateDiscretizer()
    agent = TabularQAgent()

    # 2. Load trained Q-Table (if available) and switch to pure greedy evaluation mode
    cp_file = Path(checkpoint_path)
    if cp_file.exists():
        agent.load(checkpoint_path)
    else:
        print(f"[AI Player] Note: Checkpoint '{checkpoint_path}' not found. Running with uninitialized policy.")

    agent.set_mode(training=False)

    print(os.linesep + f"[AI Player] Playing Level {level_id}: {cfg.name}")
    print("[AI Player] Press [Esc] to quit, [R] to restart episode.")

    episode = 1
    clock = pygame.time.Clock()

    while True:
        raw_state = env.reset()
        discrete_state = discretizer.discretize(raw_state)
        done = False
        steps = 0
        items_collected = 0

        while not done:
            # Render game window
            env.render()
            clock.tick(60)  # Lock to 60 FPS for viewing

            # Handle Pygame window events
            if env._screen is None:
                return
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    elif event.key == pygame.K_r:
                        restart = True
                        break

            if restart:
                break

            # 1. Agent observes discrete state and queries Q-table for best action
            action = agent.select_action(discrete_state)

            # 2. Game executes action and returns new state + event flags
            next_raw_state, events, done = env.step(action)

            # 3. Discretizer converts new physics into discrete state
            discrete_state = discretizer.discretize(next_raw_state)
            steps += 1

            if events.collected_item:
                items_collected += 1

            if done:
                if events.reached_goal:
                    print(
                        f"Episode {episode}: Hoá reached home in {steps} steps! "
                        f"(Warmth items collected: {items_collected})"
                    )
                elif events.hit_ghost:
                    print(f"Episode {episode}: Hoá was caught by a ghost at step {steps}.")
                elif events.fell_in_pit:
                    print(f"Episode {episode}: Hoá fell into a grave pit at step {steps}.")
                else:
                    print(f"Episode {episode}: Episode ended / timed out at step {steps}.")

                time.sleep(1.0)
                episode += 1


if __name__ == "__main__":
    play_autonomous(checkpoint_path="checkpoints/hybrid_q_sarsa.pkl", level_id=1)
