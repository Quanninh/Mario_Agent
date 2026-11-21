import os
import tempfile
import unittest
import numpy as np

from core.base_env import Action
from core.interfaces import BaseAgent
from agent.hybrid_q_sarsa_agent import HybridQSARSAAgent
from agent import HybridQSARSAAgent as HybridQSARSAAgentFromAgent


class TestHybridQSARSAAgent(unittest.TestCase):
    def test_implements_interface(self) -> None:
        agent = HybridQSARSAAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            HybridQSARSAAgent(beta=-0.1)
        with self.assertRaises(ValueError):
            HybridQSARSAAgent(beta=1.5)
        with self.assertRaises(ValueError):
            HybridQSARSAAgent(mode="invalid_mode")

    def test_action_selection(self) -> None:
        agent = HybridQSARSAAgent(seed=42)
        s = (1, 0, 0, 0, 0, 0, 0)
        action = agent.select_action(s)
        self.assertIn(action, list(range(Action.NUM_ACTIONS)))

    def test_linear_hybrid_update(self) -> None:
        agent = HybridQSARSAAgent(beta=0.5, mode="linear", learning_rate=0.1, discount_factor=0.9)
        s = (1, 0, 0, 0, 0, 0, 0)
        s_next = (1, 0, 1, 0, 0, 0, 0)

        agent.q_table[s_next][Action.RIGHT] = 10.0
        agent.q_table[s_next][Action.RIGHT_JUMP] = 20.0  # max

        # next_action = RIGHT (10), max = 20
        # target = 2.0 + 0.9 * [0.5 * 10 + 0.5 * 20] = 2.0 + 0.9 * 15 = 15.5
        td_err = agent.update_hybrid(
            state=s,
            action=Action.RIGHT,
            reward=2.0,
            next_state=s_next,
            next_action=Action.RIGHT,
            done=False,
        )
        self.assertAlmostEqual(td_err, 15.5, places=5)
        self.assertAlmostEqual(float(agent.q_table[s][Action.RIGHT]), 1.55, places=5)

    def test_expected_sarsa_target(self) -> None:
        agent = HybridQSARSAAgent(mode="expected", epsilon=0.2, discount_factor=0.9)
        s = (1, 0, 0, 0, 0, 0, 0)
        s_next = (1, 0, 1, 0, 0, 0, 0)

        agent.q_table[s_next] = np.array([0.0, 0.0, 0.0, 10.0, 20.0], dtype=np.float32)
        # max = 20, mean = 6.0
        # expected = (1 - 0.2) * 20 + 0.2 * 6.0 = 17.2
        # target = 2.0 + 0.9 * 17.2 = 17.48
        target = agent.compute_td_target(reward=2.0, next_state=s_next, next_action=None, done=False)
        self.assertAlmostEqual(target, 17.48, places=4)

    def test_hazard_gated_mode(self) -> None:
        agent = HybridQSARSAAgent(beta=0.5, mode="hazard_gated", discount_factor=0.9)
        # Near pit (index 4 > 0) -> pure Q-learning (max = 20)
        s_pit = (1, 0, 0, 0, 2, 0, 0)
        agent.q_table[s_pit][Action.RIGHT] = 10.0
        agent.q_table[s_pit][Action.RIGHT_JUMP] = 20.0
        target_pit = agent.compute_td_target(reward=2.0, next_state=s_pit, next_action=Action.RIGHT, done=False)
        self.assertAlmostEqual(target_pit, 2.0 + 0.9 * 20.0)

        # Safe corridor (index 4 == 0) -> blend (15)
        s_safe = (1, 0, 0, 0, 0, 0, 0)
        agent.q_table[s_safe][Action.RIGHT] = 10.0
        agent.q_table[s_safe][Action.RIGHT_JUMP] = 20.0
        target_safe = agent.compute_td_target(reward=2.0, next_state=s_safe, next_action=Action.RIGHT, done=False)
        self.assertAlmostEqual(target_safe, 2.0 + 0.9 * 15.0)

    def test_save_and_load(self) -> None:
        agent = HybridQSARSAAgent(beta=0.4, mode="hazard_gated", seed=99)
        s = (1, 0, 2, 0, 1, 0, 0)
        agent.q_table[s][Action.RIGHT] = 55.5

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "hybrid.pkl")
            json_path = os.path.join(tmpdir, "hybrid.json")

            agent.save(pkl_path)
            loaded_pkl = HybridQSARSAAgent()
            loaded_pkl.load(pkl_path)
            self.assertEqual(loaded_pkl.beta, 0.4)
            self.assertEqual(loaded_pkl.mode, "hazard_gated")
            self.assertAlmostEqual(loaded_pkl.q_table[s][Action.RIGHT], 55.5)

            agent.save(json_path)
            loaded_json = HybridQSARSAAgent()
            loaded_json.load(json_path)
            self.assertEqual(loaded_json.beta, 0.4)
            self.assertEqual(loaded_json.mode, "hazard_gated")
            self.assertAlmostEqual(loaded_json.q_table[s][Action.RIGHT], 55.5)


if __name__ == "__main__":
    unittest.main()
