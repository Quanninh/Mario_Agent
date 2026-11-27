import numpy as np
import pytest
import torch

from agent.dqn_agent import DQNAgent
from agent.dqn_network import DuelingDQN
from agent.replay_buffer import ReplayBuffer
from discretizer.vector_extractor import ContinuousStateExtractor
from env import create_env
from rewards.balanced_reward import BalancedRewardStrategy


def test_continuous_state_extractor_dimension():
    extractor = ContinuousStateExtractor()
    env = create_env(level_id=1)
    raw = env.reset()
    vec = extractor.extract(raw)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (ContinuousStateExtractor.STATE_DIM,)
    assert vec.shape == (63,)
    assert vec.dtype == np.float32
    # Verify values are in bounded normalized range
    assert np.all(vec >= -1.0)
    assert np.all(vec <= 1.0)
    env.close()


def test_continuous_state_extractor_section7_fields():
    extractor = ContinuousStateExtractor()
    env = create_env(level_id=1)
    raw = env.reset()
    # Attach Section 7 extended continuous/cyclic attributes dynamically
    raw.obstacle_dy = -40.0
    raw.obstacle_width = 32.0
    raw.pit_fire_sin = 0.5
    raw.pit_fire_cos = -0.866
    raw.ghost_float_amplitude = 25.0
    raw.ghost_float_speed = 0.1
    raw.ghost_phase_sin = -0.7
    raw.ghost_phase_cos = 0.7
    raw.ghost_dist_to_left_bound = 40.0
    raw.ghost_dist_to_right_bound = 80.0
    raw.dist_to_laser = 50.0
    raw.laser_angle_sin = 0.6
    raw.laser_angle_cos = 0.8
    raw.laser_angular_vel = 0.05
    raw.laser_length = 120.0
    raw.laser_sweep_amplitude = 1.57
    raw.wisp_ghost_dx = 100.0
    raw.wisp_ghost_dy = -50.0
    raw.wisp_cast_progress = 0.75
    raw.wisp_ticks_to_cast = 30.0
    raw.wisp_next_is_high = True
    raw.wisp_dy = -20.0
    raw.wisp_vx = -4.0
    raw.wisp_vy = -2.0
    raw.wisp_roll_ticks_left = 15
    raw.monster_type = 1
    raw.monster_is_sentry = True
    raw.monster_vy = -5.0
    raw.monster_pause_progress = 0.4
    raw.monster_speed_ratio = 0.8
    raw.monster_dist_to_pit = 60.0

    vec = extractor.extract(raw)
    assert vec.shape == (63,)
    assert np.all(vec >= -1.0)
    assert np.all(vec <= 1.0)
    env.close()


def test_dueling_dqn_forward():
    dim = ContinuousStateExtractor.STATE_DIM
    net = DuelingDQN(state_dim=dim, num_actions=6, hidden_dim=64)
    dummy = torch.randn(4, dim)
    q_vals = net(dummy)

    assert q_vals.shape == (4, 6)
    assert not torch.isnan(q_vals).any()


def test_replay_buffer_push_sample():
    dim = ContinuousStateExtractor.STATE_DIM
    buf = ReplayBuffer(capacity=100)
    s = np.zeros(dim, dtype=np.float32)
    ns = np.ones(dim, dtype=np.float32)

    for i in range(10):
        buf.push(s, 2, 1.0, ns, False)

    assert len(buf) == 10
    states, actions, rewards, next_states, dones = buf.sample(4, torch.device("cpu"))
    assert states.shape == (4, dim)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, dim)
    assert dones.shape == (4, 1)


def test_dqn_agent_step_and_update():
    dim = ContinuousStateExtractor.STATE_DIM
    agent = DQNAgent(state_dim=dim, num_actions=6, warmup_steps=5, batch_size=4, device="cpu")
    extractor = ContinuousStateExtractor()
    reward_fn = BalancedRewardStrategy()
    env = create_env(level_id=1)

    raw = env.reset()
    reward_fn.reset()
    s = extractor.extract(raw)

    losses = []
    for _ in range(15):
        a = agent.select_action(s)
        assert 0 <= a < 6
        next_raw, events, done = env.step(a)
        r = reward_fn.calculate_reward(raw, a, next_raw, events, done)
        ns = extractor.extract(next_raw)
        loss = agent.update(s, a, r, ns, done)
        losses.append(loss)
        s = ns
        raw = next_raw
        if done:
            raw = env.reset()
            reward_fn.reset()
            s = extractor.extract(raw)

    # After warmup (step >= 5), updates should produce positive scalar loss
    assert any(l > 0.0 for l in losses)
    env.close()


def test_dqn_agent_save_load(tmp_path):
    dim = ContinuousStateExtractor.STATE_DIM
    save_path = tmp_path / "test_dqn.pt"
    agent1 = DQNAgent(state_dim=dim, num_actions=6, epsilon=0.42, device="cpu")
    agent1.save(save_path)

    agent2 = DQNAgent(state_dim=dim, num_actions=6, device="cpu")
    agent2.load(save_path)

    assert np.isclose(agent2.epsilon, 0.42)
    # Check that weights match
    p1 = next(agent1.policy_net.parameters()).data.numpy()
    p2 = next(agent2.policy_net.parameters()).data.numpy()
    np.testing.assert_allclose(p1, p2)


def test_balanced_reward_consistent_with_raw_state():
    import math
    from core.base_env import Action, GameEvents, RawState

    strategy = BalancedRewardStrategy(
        unnecessary_jump_penalty=-1.5,
        laser_danger_penalty=-3.0,
        wisp_danger_penalty=-3.0,
        step_penalty=-0.1,
        use_max_x=False,
    )

    def make_base_state(**kwargs):
        defaults = dict(
            player_x=100.0,
            player_y=288.0,
            velocity_x=0.0,
            velocity_y=0.0,
            is_grounded=True,
            can_double_jump=False,
            is_crouching=False,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            pit_fire_state=0,
            dist_to_ghost=-1.0,
            ghost_y=0.0,
            dist_to_item=-1.0,
            dist_to_laser=-1.0,
            dist_to_wisp=-1.0,
            wisp_dir=0,
            dist_to_monster=-1.0,
            monster_y=0.0,
            dist_to_goal=500.0,
        )
        defaults.update(kwargs)
        return RawState(**defaults)

    # 1. Floating overhead obstacle should NOT exempt unnecessary jump penalty
    s_obs = make_base_state(dist_to_obstacle=20.0, obstacle_height=20.0, obstacle_dy=-68.0)
    r_obs = strategy.calculate_reward(s_obs, Action.JUMP, s_obs, GameEvents(dx=0.0), False)
    assert np.isclose(r_obs, -1.6)  # -0.1 step - 1.5 unnecessary jump

    # 2. Continuous LaserGhost closing window even when dist_to_laser == -1.0 (pointing upward)
    # angle = -0.1 rad -> sin=-0.0998, cos=0.995; rotating +0.05 rad/tick crosses 0 into floor within 8 ticks
    ang = -0.1
    s_laser = make_base_state(
        dist_to_laser=-1.0,
        laser_ghost_dx=40.0,
        laser_length=90.0,
        laser_angle=ang,
        laser_angle_sin=math.sin(ang),
        laser_angle_cos=math.cos(ang),
        laser_angular_vel=0.05,
    )
    r_laser = strategy.calculate_reward(s_laser, Action.RIGHT, s_laser, GameEvents(dx=0.0), False)
    assert np.isclose(r_laser, -3.1)  # -0.1 step - 3.0 laser danger

    # 3. Distant WispGhost charging (>150px away) should NOT penalize RIGHT_JUMP
    s_wisp_far = make_base_state(
        dist_to_pit=20.0,
        wisp_ghost_charging=True,
        wisp_charging=False,
        wisp_ghost_dx=300.0,
    )
    r_wisp_far = strategy.calculate_reward(s_wisp_far, Action.RIGHT_JUMP, s_wisp_far, GameEvents(dx=0.0), False)
    assert np.isclose(r_wisp_far, -0.1)  # only step penalty
