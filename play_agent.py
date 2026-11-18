import time
import pygame
import os
from env import create_env, get_level
from discretizer.state_discretizer import MarioStateDiscretizer
from agent.tabular_q_agent import TabularQAgent


def play_autonomous(checkpoint_path: str = "checkpoints/q_table_best.pkl", level_id: int = 1):
    # 1. Instantiate Environment, Discretizer, and Agent
    cfg = get_level(level_id)
    env = create_env(level_id)
    discretizer = MarioStateDiscretizer()
    agent = TabularQAgent()

    # 2. Load trained Q-Table and switch to pure greedy evaluation mode (epsilon = 0.0)
    agent.load_q_table(checkpoint_path)
    agent.set_mode(training=False)

    print(os.linesep + f"[AI Player] Playing Level {level_id}: {cfg.name}")
    print("[AI Player] Press [Esc] to quit, [R] to restart episode.")

    episode = 1
    clock = pygame.time.Clock()

    while True:
        raw_state = env.reset()
        discrete_state = discretizer.discretize(raw_state)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Render game window
            env.render()
            clock.tick(60)  # Lock to 60 FPS for human viewing

            # Handle Pygame exit events
            if env._screen is None:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return

            # --- CONNECT AGENT WITH GAME ---
            # 1. Agent observes discrete state and queries Q-table for best action
            action = agent.select_action(discrete_state)

            # 2. Game executes action and returns new state + event flags
            next_raw_state, events, done = env.step(action)

            # 3. Discretizer converts new physics into discrete state
            discrete_state = discretizer.discretize(next_raw_state)
            steps += 1

            if done:
                if events.reached_goal:
                    print(f"Episode {episode}: AI reached flagpole in {steps} steps!")
                elif events.fell_in_pit:
                    print(f"Episode {episode}: AI fell into pit at step {steps}.")
                else:
                    print(f"Episode {episode}: Timed out at step {steps}.")

                time.sleep(1.0)  # Pause briefly before next run
                episode += 1


if __name__ == "__main__":
    play_autonomous(checkpoint_path="checkpoints/q_table_best.pkl", level_id=1)