import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from agent.dqn_agent import DQNAgent
from discretizer.vector_extractor import ContinuousStateExtractor
from env import create_env
from rewards.balanced_reward import BalancedRewardStrategy
from training.trainer import resolve_map_seed


def train_dqn_curriculum(
    levels: List[int],
    episodes_per_level: int = 250,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 64,
    buffer_capacity: int = 100_000,
    warmup_steps: int = 500,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    frame_skip: int = 1,
    checkpoint_dir: str = "checkpoints/dqn",
    pretrained_path: Optional[str] = None,
    device: Optional[str] = None,
    map_seed: Optional[int] = None,
    map_seed_interval: int = 0,
    seed: Optional[int] = None,
    eval_episodes: int = 0,
) -> Dict:
    """
    Trains a Dueling Double DQN agent across a curriculum of game levels.
    Evaluates on fixed handcrafted maps when eval_episodes > 0.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    fallback_seed = 42 if seed is None else int(seed)

    extractor = ContinuousStateExtractor()
    reward_fn = BalancedRewardStrategy()

    agent = DQNAgent(
        state_dim=ContinuousStateExtractor.STATE_DIM,
        num_actions=6,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        training=True,
        device=device,
        seed=seed,
    )

    if pretrained_path is not None and Path(pretrained_path).exists():
        print(f"[PRETRAINED] Loading initial model weights from: {pretrained_path}")
        agent.load(pretrained_path)
        # Re-warm exploration for unfamiliar curriculum stages
        agent.epsilon = max(0.25, agent.epsilon)

    print(f"[{'='*30} STARTING DQN CURRICULUM TRAINING {'='*30}]")
    print(f"Device: {agent.device} | Levels: {levels} | Episodes/Level: {episodes_per_level}")
    print(
        f"Frame skip: {frame_skip} | Batch size: {batch_size} | LR: {learning_rate} | "
        f"MapSeed: {map_seed if map_seed is not None else (f'random(seed={fallback_seed})' if map_seed_interval > 0 else None)} | "
        f"MapRotateInterval: {map_seed_interval}"
    )
    if map_seed is not None or map_seed_interval > 0:
        if any(lvl > 7 for lvl in levels):
            print("[Note] Procedural map seeds apply to Levels 1-7 only; Level 8 remains a fixed handcrafted layout.")
    print("-" * 85)

    global_episode = 0
    total_episodes = len(levels) * episodes_per_level
    start_time = time.time()
    history = []
    eval_summary: Dict[int, Dict[str, float]] = {}

    for level_id in levels:
        current_map_seed = resolve_map_seed(
            level_id, 1, map_seed, map_seed_interval, fallback_seed=fallback_seed
        )
        env = create_env(level_id=level_id, seed=current_map_seed)
        level_wins = []
        best_win_rate = 0.0

        # Adjust exploration when advancing stages if needed
        if level_id > levels[0] and agent.epsilon < 0.25:
            agent.epsilon = 0.25

        try:
            for ep in range(1, episodes_per_level + 1):
                global_episode += 1
                desired_map_seed = resolve_map_seed(
                    level_id, ep, map_seed, map_seed_interval, fallback_seed=fallback_seed
                )
                if desired_map_seed != current_map_seed:
                    env.close()
                    current_map_seed = desired_map_seed
                    env = create_env(level_id=level_id, seed=current_map_seed)
                    print(
                        f"[DQN] Level {level_id} ep {ep}: rotated map seed -> {current_map_seed}"
                    )

                if hasattr(reward_fn, "reset"):
                    reward_fn.reset()
                raw_state = env.reset()
                state_vec = extractor.extract(raw_state)
                ep_reward = 0.0
                ep_loss = 0.0
                steps = 0
                loss_count = 0
                done = False
                events = None

                while not done:
                    action = agent.select_action(state_vec)

                    # Frame skipping: repeat chosen action for frame_skip ticks
                    skip_reward = 0.0
                    for _ in range(frame_skip):
                        next_raw, events, done = env.step(action)
                        step_r = reward_fn.calculate_reward(raw_state, action, next_raw, events, done)
                        skip_reward += step_r
                        raw_state = next_raw
                        steps += 1
                        if done:
                            break

                    next_vec = extractor.extract(next_raw)
                    loss = agent.update(state_vec, action, skip_reward, next_vec, done)
                    if loss > 0.0:
                        ep_loss += loss
                        loss_count += 1

                    ep_reward += skip_reward
                    state_vec = next_vec

                agent.decay_epsilon()
                win = 1 if (events is not None and getattr(events, "reached_goal", False)) else 0
                level_wins.append(win)
                rolling_50 = float(np.mean(level_wins[-50:])) * 100.0
                avg_loss = ep_loss / max(1, loss_count)

                history.append({
                    "global_episode": global_episode,
                    "level_id": level_id,
                    "level_episode": ep,
                    "map_seed": current_map_seed,
                    "reward": ep_reward,
                    "steps": steps,
                    "win": win,
                    "rolling_win_rate_50": rolling_50,
                    "loss": avg_loss,
                    "epsilon": agent.epsilon,
                })

                # Save best checkpoint for current level
                if len(level_wins) >= 20 and rolling_50 > best_win_rate:
                    best_win_rate = rolling_50
                    agent.save(ckpt_path / f"dqn_level_{level_id}_best.pt")

                # Progress logging
                if ep == 1 or ep % 25 == 0 or ep == episodes_per_level:
                    elapsed = time.time() - start_time
                    pct = (global_episode / total_episodes) * 100.0
                    seed_str = f" | MapSeed: {current_map_seed}" if current_map_seed is not None else ""
                    print(
                        f"[{pct:5.1f}%] Global Ep {global_episode:4d}/{total_episodes:4d} | "
                        f"Lvl {level_id} ({ep:3d}/{episodes_per_level:3d}) | "
                        f"Win50: {rolling_50:5.1f}% | R: {ep_reward:+7.1f} | "
                        f"Loss: {avg_loss:6.4f} | eps: {agent.epsilon:.3f} | "
                        f"Buffer: {len(agent.replay_buffer):5d}{seed_str} | Time: {int(elapsed)}s"
                    )
        finally:
            env.close()

        if eval_episodes > 0:
            eval_env = create_env(level_id=level_id, seed=None)
            prev_training = agent.training
            agent.set_mode(False)
            eval_wins = 0
            eval_rewards: List[float] = []
            try:
                for _ in range(eval_episodes):
                    if hasattr(reward_fn, "reset"):
                        reward_fn.reset()
                    e_raw = eval_env.reset()
                    e_vec = extractor.extract(e_raw)
                    e_done = False
                    e_ret = 0.0
                    e_events = None
                    while not e_done:
                        e_act = agent.select_action(e_vec)
                        e_next, e_events, e_done = eval_env.step(e_act)
                        e_ret += reward_fn.calculate_reward(e_raw, e_act, e_next, e_events, e_done)
                        e_raw = e_next
                        e_vec = extractor.extract(e_raw)
                    if e_events is not None and getattr(e_events, "reached_goal", False):
                        eval_wins += 1
                    eval_rewards.append(e_ret)
            finally:
                eval_env.close()
                agent.set_mode(prev_training)
            eval_wr = round((eval_wins / eval_episodes) * 100.0, 2)
            eval_avg_r = round(float(np.mean(eval_rewards)), 2) if eval_rewards else 0.0
            eval_summary[level_id] = {"win_rate": eval_wr, "avg_reward": eval_avg_r}
            print(
                f"[DQN Eval | Fixed Lvl {level_id} ({eval_episodes} eps)] "
                f"WinRate: {eval_wr:.1f}% | Avg R: {eval_avg_r:+.1f}"
            )

    # Save final model
    agent.save(ckpt_path / "dqn_final.pt")
    print(f"\n[OK] Training completed! Model saved to {ckpt_path / 'dqn_final.pt'}")
    return {"history": history, "agent": agent, "eval_summary": eval_summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dueling Double DQN Agent")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4], help="Levels to train (e.g. 1 2 3 4)")
    parser.add_argument("--episodes", type=int, default=150, help="Episodes per level")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skipping (1 or 2)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dqn", help="Checkpoint output folder")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained .pt checkpoint to resume/warm-start from")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (also base RNG seed for random map rotation)")
    parser.add_argument("--map-seed", type=int, default=None, help="Starting procedural map seed for Levels 1-7 (None = fixed handcrafted maps)")
    parser.add_argument("--map-seed-interval", type=int, default=0, help="Rotate to a new random procedural map seed every N episodes per level on Levels 1-7 (e.g. 3000 or 5000)")
    parser.add_argument("--eval-episodes", type=int, default=0, help="Optional greedy evaluation episodes on fixed handcrafted levels after each stage")
    args = parser.parse_args()

    train_dqn_curriculum(
        levels=args.levels,
        episodes_per_level=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_path=args.pretrained,
        device=args.device,
        map_seed=args.map_seed,
        map_seed_interval=args.map_seed_interval,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
    )
