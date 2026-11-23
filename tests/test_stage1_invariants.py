import unittest
from core.base_env import Action, RawState, GameEvents
from discretizer.state_discretizer import MarioStateDiscretizer
from rewards.balanced_reward import BalancedRewardStrategy
from rewards.dense_reward import DenseRewardStrategy
from rewards.sparse_reward import SparseRewardStrategy


class TestStage1Invariants(unittest.TestCase):
    def setUp(self):
        self.discretizer = MarioStateDiscretizer()
        self.balanced = BalancedRewardStrategy()
        self.dense = DenseRewardStrategy()
        self.sparse = SparseRewardStrategy()

    def _make_state(
        self,
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
        dist_to_ghost=20.0,
        ghost_y=0.0,
        dist_to_item=-1.0,
        dist_to_laser=-1.0,
        dist_to_wisp=-1.0,
        wisp_dir=0,
        dist_to_monster=-1.0,
        monster_y=0.0,
        dist_to_goal=500.0,
    ):
        return RawState(
            player_x=player_x,
            player_y=player_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            is_grounded=is_grounded,
            can_double_jump=can_double_jump,
            is_crouching=is_crouching,
            dist_to_obstacle=dist_to_obstacle,
            obstacle_height=obstacle_height,
            dist_to_pit=dist_to_pit,
            pit_fire_state=pit_fire_state,
            dist_to_ghost=dist_to_ghost,
            ghost_y=ghost_y,
            dist_to_item=dist_to_item,
            dist_to_laser=dist_to_laser,
            dist_to_wisp=dist_to_wisp,
            wisp_dir=wisp_dir,
            dist_to_monster=dist_to_monster,
            monster_y=monster_y,
            dist_to_goal=dist_to_goal,
        )

    def test_discretizer_high_ghosts_on_floor(self):
        """All floor + high-ghost cases across full sinusoidal bobbing range return ghost_alt_bin == 1."""
        high_ghost_specs = [
            # (name, base_y, amp)
            ("L4 Ghost 2", 320.0 - 84.0, 8.0),
            ("L5 Ghost 2", 320.0 - 82.0, 4.0),
            ("L5 Ghost 3", 320.0 - 86.0, 13.0),
        ]
        for name, base_y, amp in high_ghost_specs:
            for sin_val in [-1.0, 0.0, 1.0]:
                gy = base_y + sin_val * amp
                st = self._make_state(player_y=288.0, dist_to_ghost=20.0, ghost_y=gy)
                d_state = self.discretizer.discretize(st)
                # (grounded, vy, obs_dist, obs_h, pit_dist, ghost_dist, ghost_alt)
                ghost_alt_bin = d_state[6]
                self.assertEqual(
                    ghost_alt_bin,
                    1,
                    f"Failed for {name} with sin={sin_val} (ghost_y={gy}): expected ghost_alt_bin=1, got {ghost_alt_bin}",
                )

    def test_discretizer_low_ghosts_on_floor(self):
        """All floor + low-ghost cases return ghost_alt_bin == 0."""
        low_ghost_specs = [
            # (name, base_y, amp)
            ("L4 Ghost 1", 320.0 - 40.0, 8.0),
            ("L5 Ghost 1", 320.0 - 38.0, 6.0),
            ("L5 Ghost 4", 320.0 - 40.0, 7.0),
        ]
        for name, base_y, amp in low_ghost_specs:
            for sin_val in [-1.0, 0.0, 1.0]:
                gy = base_y + sin_val * amp
                st = self._make_state(player_y=288.0, dist_to_ghost=20.0, ghost_y=gy)
                d_state = self.discretizer.discretize(st)
                ghost_alt_bin = d_state[6]
                self.assertEqual(
                    ghost_alt_bin,
                    0,
                    f"Failed for {name} with sin={sin_val} (ghost_y={gy}): expected ghost_alt_bin=0, got {ghost_alt_bin}",
                )

    def test_discretizer_tombstone_adjacent_high_ghosts(self):
        """All tombstone-top + adjacent high-ghost cases return ghost_alt_bin == 0."""
        tombstone_specs = [
            # (name, player_y, base_y, amp)
            ("L4 Obs 3 top + L4 Ghost 2", 228.0, 320.0 - 84.0, 8.0),
            ("L5 Obs 2 top + L5 Ghost 2", 228.0, 320.0 - 82.0, 4.0),
            ("L5 Obs 3 top + L5 Ghost 3", 236.0, 320.0 - 86.0, 13.0),
        ]
        for name, py, base_y, amp in tombstone_specs:
            for sin_val in [-1.0, 0.0, 1.0]:
                gy = base_y + sin_val * amp
                st = self._make_state(player_y=py, dist_to_ghost=20.0, ghost_y=gy)
                d_state = self.discretizer.discretize(st)
                ghost_alt_bin = d_state[6]
                self.assertEqual(
                    ghost_alt_bin,
                    0,
                    f"Failed for {name} with sin={sin_val} (player_y={py}, ghost_y={gy}): expected ghost_alt_bin=0, got {ghost_alt_bin}",
                )

    def test_discretizer_cardinality_and_length(self):
        """State tuple length remains 12 and max cardinality includes all hazards."""
        st = self._make_state()
        d_state = self.discretizer.discretize(st)
        self.assertEqual(len(d_state), 12)

    def test_balanced_reward_round_trip(self):
        """LEFT + RIGHT round trip on open ground nets strictly -0.20."""
        prev_st = self._make_state(player_x=100.0)
        next_left = self._make_state(player_x=96.0)
        events_left = GameEvents(dx=-4.0)

        r_left = self.balanced.calculate_reward(
            prev_st, Action.LEFT, next_left, events_left, done=False
        )
        self.assertAlmostEqual(r_left, 1.0 * (-4.0) - 0.1, places=5)

        next_right = self._make_state(player_x=100.0)
        events_right = GameEvents(dx=4.0)
        r_right = self.balanced.calculate_reward(
            next_left, Action.RIGHT, next_right, events_right, done=False
        )
        self.assertAlmostEqual(r_right, 1.0 * 4.0 - 0.1, places=5)

        net_round_trip = r_left + r_right
        self.assertAlmostEqual(
            net_round_trip,
            -0.20,
            places=5,
            msg=f"Net round trip should be -0.20, got {net_round_trip}",
        )

    def test_balanced_jump_under_high_ghost_penalized(self):
        """Grounded JUMP under L4 Ghost 2 triggers unnecessary-jump penalty."""
        # Floor: player_y=288.0, L4 Ghost 2 base_y=236.0 (overhead cleared)
        prev_st = self._make_state(
            player_y=288.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=236.0,
        )
        next_st = self._make_state(player_y=288.0, is_grounded=True)
        events = GameEvents(dx=0.0)

        # Reward with STAY (dx=0, step=-0.1)
        r_stay = self.balanced.calculate_reward(
            prev_st, Action.STAY, next_st, events, done=False
        )
        # Reward with JUMP (dx=0, step=-0.1 + unnecessary_jump_penalty)
        r_jump = self.balanced.calculate_reward(
            prev_st, Action.JUMP, next_st, events, done=False
        )

        expected_diff = self.balanced.unnecessary_jump_penalty
        self.assertAlmostEqual(
            r_jump - r_stay,
            expected_diff,
            places=5,
            msg="Jumping under overhead high ghost should be penalized by unnecessary_jump_penalty",
        )

    def test_balanced_jump_on_tombstone_eye_level_exempted(self):
        """Grounded JUMP on top of L4 Obs 3 at eye level with L4 Ghost 2 is exempted."""
        # On top of L4 Obs 3: player_y=228.0, L4 Ghost 2 base_y=236.0 (NOT overhead cleared)
        prev_st = self._make_state(
            player_y=228.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=236.0,
        )
        next_st = self._make_state(player_y=228.0, is_grounded=True)
        events = GameEvents(dx=0.0)

        r_stay = self.balanced.calculate_reward(
            prev_st, Action.STAY, next_st, events, done=False
        )
        r_jump = self.balanced.calculate_reward(
            prev_st, Action.JUMP, next_st, events, done=False
        )

        self.assertAlmostEqual(
            r_jump,
            r_stay,
            places=5,
            msg="Jumping on top of tombstone at eye level with ghost must be exempted (r_jump == r_stay)",
        )

    def test_dense_and_sparse_unchanged(self):
        """DenseRewardStrategy and SparseRewardStrategy calculate_reward behavior is untouched."""
        prev_st = self._make_state(player_x=100.0)
        next_st = self._make_state(player_x=104.0)
        events = GameEvents(dx=4.0)

        r_dense = self.dense.calculate_reward(prev_st, Action.RIGHT, next_st, events, False)
        self.assertIsInstance(r_dense, float)

        r_sparse = self.sparse.calculate_reward(prev_st, Action.RIGHT, next_st, events, False)
        self.assertEqual(r_sparse, 0.0)


if __name__ == "__main__":
    unittest.main()
