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
    assert vec.shape == (22,)
    assert vec.dtype == np.float32
    # Verify values are in bounded normalized range
    assert np.all(vec >= -1.0)
    assert np.all(vec <= 1.0)
    env.close()


def test_dueling_dqn_forward():
    net = DuelingDQN(state_dim=22, num_actions=6, hidden_dim=64)
    dummy = torch.randn(4, 22)
    q_vals = net(dummy)

    assert q_vals.shape == (4, 6)
    assert not torch.isnan(q_vals).any()


def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100)
    s = np.zeros(22, dtype=np.float32)
    ns = np.ones(22, dtype=np.float32)

    for i in range(10):
        buf.push(s, 2, 1.0, ns, False)

    assert len(buf) == 10
    states, actions, rewards, next_states, dones = buf.sample(4, torch.device("cpu"))
    assert states.shape == (4, 22)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 22)
    assert dones.shape == (4, 1)


def test_dqn_agent_step_and_update():
    agent = DQNAgent(state_dim=22, num_actions=6, warmup_steps=5, batch_size=4, device="cpu")
    extractor = ContinuousStateExtractor()
    reward_fn = BalancedRewardStrategy()
    env = create_env(level_id=1)

    raw = env.reset()
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
            s = extractor.extract(raw)

    # After warmup (step >= 5), updates should produce positive scalar loss
    assert any(l > 0.0 for l in losses)
    env.close()


def test_dqn_agent_save_load(tmp_path):
    save_path = tmp_path / "test_dqn.pt"
    agent1 = DQNAgent(state_dim=22, num_actions=6, epsilon=0.42, device="cpu")
    agent1.save(save_path)

    agent2 = DQNAgent(state_dim=22, num_actions=6, device="cpu")
    agent2.load(save_path)

    assert np.isclose(agent2.epsilon, 0.42)
    # Check that weights match
    p1 = next(agent1.policy_net.parameters()).data.numpy()
    p2 = next(agent2.policy_net.parameters()).data.numpy()
    np.testing.assert_allclose(p1, p2)
