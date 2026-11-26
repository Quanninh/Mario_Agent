import os
import tempfile
from typing import Optional
import unittest

from core.base_env import Action, GameEvents, RawState
from core.interfaces import BaseAgent, BaseDiscretizer, BaseRewardStrategy
from discretizer import MarioStateDiscretizer, PlayerStateDiscretizer
from rewards import BalancedRewardStrategy, DenseRewardStrategy, SparseRewardStrategy
from agent import TabularQAgent


def make_raw_state(
    player_x: float = 50.0,
    player_y: float = 288.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    is_grounded: bool = True,
    can_double_jump: bool = False,
    is_crouching: bool = False,
    dist_to_obstacle: float = -1.0,
    obstacle_height: float = 0.0,
    dist_to_pit: float = -1.0,
    pit_fire_state: int = 0,
    dist_to_ghost: float = -1.0,
    ghost_y: float = 0.0,
    dist_to_item: float = -1.0,
    dist_to_laser: float = -1.0,
    dist_to_wisp: float = -1.0,
    wisp_dir: int = 0,
    dist_to_monster: float = -1.0,
    monster_y: float = 0.0,
    dist_to_goal: float = 500.0,
    ghost_vx: float = 0.0,
    laser_angle: float = -1.0,
    laser_rot_speed: float = 0.0,
    wisp_vx: float = 0.0,
    wisp_vy: float = 0.0,
    wisp_y: float = 0.0,
    wisp_charging: bool = False,
    monster_vx: float = 0.0,
    monster_vy: float = 0.0,
    monster_type: Optional[str] = None,
) -> RawState:
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
        ghost_vx=ghost_vx,
        laser_angle=laser_angle,
        laser_rot_speed=laser_rot_speed,
        wisp_vx=wisp_vx,
        wisp_vy=wisp_vy,
        wisp_y=wisp_y,
        wisp_charging=wisp_charging,
        monster_vx=monster_vx,
        monster_vy=monster_vy,
        monster_type=monster_type,
    )


class TestDiscretizer(unittest.TestCase):
    def setUp(self) -> None:
        self.discretizer = MarioStateDiscretizer()

    def test_implements_interface(self) -> None:
        self.assertIsInstance(self.discretizer, BaseDiscretizer)
        self.assertTrue(issubclass(PlayerStateDiscretizer, BaseDiscretizer))

    def test_discretize_structure_and_types(self) -> None:
        raw = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=50.0,
            obstacle_height=48.0,
            dist_to_pit=60.0,
            dist_to_ghost=50.0,
            ghost_y=280.0,
            dist_to_item=30.0,
            dist_to_goal=500.0,
        )
        state = self.discretizer.discretize(raw)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 15)
        self.assertTrue(all(isinstance(x, int) for x in state))

    def test_legacy_attribute_compatibility(self) -> None:
        # If legacy state has mario_y instead of player_y, it should not crash
        class LegacyState:
            mario_y = 288.0
            is_grounded = True
            velocity_y = 0.0
            dist_to_obstacle = 20.0
            obstacle_height = 40.0
            dist_to_pit = -1.0
            dist_to_ghost = -1.0
            ghost_y = 0.0

        state = self.discretizer.discretize(LegacyState())
        self.assertEqual(len(state), 15)

    def test_ghost_dist_and_alt_bins(self) -> None:
        # Low ghost (jump over)
        raw_low = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=280.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_low = self.discretizer.discretize(raw_low)
        self.assertEqual(s_low[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_low[6], 0)  # ghost_alt_bin: LOW

        # High ghost (walk under)
        raw_high = make_raw_state(
            player_x=50.0,
            player_y=288.0,
            velocity_y=0.0,
            is_grounded=True,
            dist_to_obstacle=-1.0,
            obstacle_height=0.0,
            dist_to_pit=-1.0,
            dist_to_ghost=20.0,
            ghost_y=234.0,
            dist_to_item=-1.0,
            dist_to_goal=400.0,
        )
        s_high = self.discretizer.discretize(raw_high)
        self.assertEqual(s_high[5], 2)  # ghost_dist_bin: CLOSE
        self.assertEqual(s_high[6], 1)  # ghost_alt_bin: HIGH

    def test_wisp_and_double_jump_bins(self) -> None:
        # Double jump armed
        raw_dj = make_raw_state(can_double_jump=True, is_grounded=False)
        s_dj = self.discretizer.discretize(raw_dj)
        self.assertEqual(s_dj[10], 1)  # double_jump_bin: 1

        # Wisp ahead within threat distance
        raw_wisp_ahead = make_raw_state(dist_to_wisp=30.0, wisp_dir=1)
        s_wisp_ahead = self.discretizer.discretize(raw_wisp_ahead)
        self.assertEqual(s_wisp_ahead[8], 1)  # wisp_bin: INCOMING_AHEAD

        # Wisp behind
        raw_wisp_behind = make_raw_state(dist_to_wisp=30.0, wisp_dir=-1)
        s_wisp_behind = self.discretizer.discretize(raw_wisp_behind)
        self.assertEqual(s_wisp_behind[8], 2)  # wisp_bin: CHASING_BEHIND

        # Wisp far
        raw_wisp_far = make_raw_state(dist_to_wisp=100.0, wisp_dir=1)
        s_wisp_far = self.discretizer.discretize(raw_wisp_far)
        self.assertEqual(s_wisp_far[8], 0)  # wisp_bin: SAFE

        # Monster approaching
        raw_mon_near = make_raw_state(dist_to_monster=50.0)
        s_mon_near = self.discretizer.discretize(raw_mon_near)
        self.assertEqual(s_mon_near[11], 1)  # monster_bin: 1

        # Monster far / none
        raw_mon_far = make_raw_state(dist_to_monster=150.0)
        s_mon_far = self.discretizer.discretize(raw_mon_far)
        self.assertEqual(s_mon_far[11], 0)  # monster_bin: 0

    def test_laser_and_monster_bins_with_rotation_and_velocity(self) -> None:
        # Laser pointing skyward (angle ~ 4.71 rad / 270 deg) -> laser_bin: 1 (SAFE_WINDOW)
        raw_laser_sky = make_raw_state(dist_to_laser=40.0, laser_angle=4.71)
        s_laser_sky = self.discretizer.discretize(raw_laser_sky)
        self.assertEqual(s_laser_sky[7], 1)

        # Laser sweeping floor (angle ~ 1.57 rad / 90 deg) -> laser_bin: 2 (GROUND_DANGER)
        raw_laser_floor = make_raw_state(dist_to_laser=40.0, laser_angle=1.57)
        s_laser_floor = self.discretizer.discretize(raw_laser_floor)
        self.assertEqual(s_laser_floor[7], 2)

        # Cuong Thi airborne hopping (vy < -1.0) -> monster_bin: 1
        raw_cuong_thi_hop = make_raw_state(dist_to_monster=50.0, monster_vy=-4.0, monster_type="cuong_thi")
        s_hop = self.discretizer.discretize(raw_cuong_thi_hop)
        self.assertEqual(s_hop[11], 1)

        # Cuong Thi paused on ground (vy == 0.0) -> monster_bin: 2
        raw_cuong_thi_ground = make_raw_state(dist_to_monster=50.0, monster_vy=0.0, monster_type="cuong_thi")
        s_ground = self.discretizer.discretize(raw_cuong_thi_ground)
        self.assertEqual(s_ground[11], 2)

        # Quy Nhap Trang high-speed ground sprint (vx < -2.0) -> monster_bin: 3
        raw_quy_nhap_trang = make_raw_state(dist_to_monster=50.0, monster_vx=-4.2, monster_type="quy_nhap_trang")
        s_sprint = self.discretizer.discretize(raw_quy_nhap_trang)
        self.assertEqual(s_sprint[11], 3)

        # Laser skyward but rotating fast clockwise toward ground sweep -> laser_bin: 3 (CLOSING_WINDOW)
        raw_laser_closing = make_raw_state(dist_to_laser=40.0, laser_angle=6.10, laser_rot_speed=0.05)
        self.assertEqual(self.discretizer.discretize(raw_laser_closing)[7], 3)

        # Laser skyward and rotating away from ground sweep -> laser_bin: 1 (SAFE_WINDOW)
        raw_laser_opening = make_raw_state(dist_to_laser=40.0, laser_angle=3.30, laser_rot_speed=0.05)
        self.assertEqual(self.discretizer.discretize(raw_laser_opening)[7], 1)

        # Wisp high overhead -> wisp_bin: 3 (INCOMING_HIGH)
        raw_wisp_high = make_raw_state(player_y=288.0, dist_to_wisp=30.0, wisp_dir=1, wisp_y=240.0)
        self.assertEqual(self.discretizer.discretize(raw_wisp_high)[8], 3)

        # WispGhost charging ahead with no active wisp in range -> wisp_bin: 4 (CHARGING)
        raw_wisp_charging = make_raw_state(dist_to_wisp=-1.0, wisp_charging=True)
        self.assertEqual(self.discretizer.discretize(raw_wisp_charging)[8], 4)


class TestRewards(unittest.TestCase):
    def setUp(self) -> None:
        self.s1 = make_raw_state(player_x=50.0, player_y=288.0, dist_to_goal=500.0)
        self.s2 = make_raw_state(player_x=54.0, player_y=288.0, dist_to_goal=496.0)

    def test_balanced_reward_ghost_and_item(self) -> None:
        strategy = BalancedRewardStrategy(
            goal_bonus=200.0,
            pit_penalty=-75.0,
            ghost_penalty=-75.0,
            item_bonus=25.0,
            step_penalty=-0.1,
            use_max_x=False,
        )
        self.assertIsInstance(strategy, BaseRewardStrategy)

        # Normal forward step
        ev_norm = GameEvents(dx=4.0)
        r_norm = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_norm, False)
        self.assertAlmostEqual(r_norm, 3.9)  # 4.0 - 0.1

        # Death by ghost: MUST be severely penalized
        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        r_ghost = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True)
        self.assertAlmostEqual(r_ghost, -71.1)  # 4.0 - 75.0 - 0.1

        # Death by pit: MUST be severely penalized
        ev_pit = GameEvents(dx=4.0, fell_in_pit=True)
        r_pit = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_pit, True)
        self.assertAlmostEqual(r_pit, -71.1)

        # Death by monster: MUST be severely penalized (uses default death_penalty=-100.0)
        ev_mon = GameEvents(dx=4.0, hit_monster=True)
        r_mon = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_mon, True)
        self.assertAlmostEqual(r_mon, -96.1)  # 4.0 - 100.0 - 0.1

        # Unified died event flag
        ev_died = GameEvents(dx=4.0, died=True)
        r_died = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_died, True)
        self.assertAlmostEqual(r_died, -96.1)

        # Obstacle collision: no penalty
        ev_obs = GameEvents(dx=0.0, hit_obstacle=True)
        r_obs = strategy.calculate_reward(self.s1, Action.RIGHT, self.s1, ev_obs, False)
        self.assertAlmostEqual(r_obs, -0.1)  # 0.0 - 0.1

        # Warmth item collection bonus
        ev_item = GameEvents(dx=4.0, collected_item=True)
        r_item = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False)
        self.assertAlmostEqual(r_item, 28.9)  # 4.0 + 25.0 - 0.1

        # Goal reached
        ev_goal = GameEvents(dx=4.0, reached_goal=True)
        r_goal = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_goal, True)
        self.assertAlmostEqual(r_goal, 203.9)  # 4.0 + 200.0 - 0.1

    def test_balanced_reward_jump_exemption_and_spam_penalty(self) -> None:
        strategy = BalancedRewardStrategy(unnecessary_jump_penalty=-15.0)

        # Baseline: walking on flat open ground
        s_open = make_raw_state(is_grounded=True)
        ev = GameEvents(dx=4.0)
        r_walk = strategy.calculate_reward(s_open, Action.RIGHT, s_open, ev, False)

        # 1. Unprovoked jump on open ground -> MUST incur -15.0 penalty
        r_unprovoked_jump = strategy.calculate_reward(s_open, Action.RIGHT_JUMP, s_open, ev, False)
        self.assertAlmostEqual(r_unprovoked_jump - r_walk, -15.0)

        # 2. Jump when monster is approaching at 60px (<= 75px threshold) -> EXEMPT
        s_monster = make_raw_state(is_grounded=True, dist_to_monster=60.0)
        r_jump_monster = strategy.calculate_reward(s_monster, Action.RIGHT_JUMP, s_monster, ev, False)
        self.assertAlmostEqual(r_jump_monster, r_walk)

        # 3. Jump when monster is far at 100px (> 75px threshold) -> PENALIZED
        s_monster_far = make_raw_state(is_grounded=True, dist_to_monster=100.0)
        r_jump_monster_far = strategy.calculate_reward(s_monster_far, Action.RIGHT_JUMP, s_monster_far, ev, False)
        self.assertAlmostEqual(r_jump_monster_far - r_walk, -15.0)

        # 4. Jump when wisp is incoming at 55px (<= 65px threshold) -> EXEMPT
        s_wisp = make_raw_state(is_grounded=True, dist_to_wisp=55.0, wisp_dir=1)
        r_jump_wisp = strategy.calculate_reward(s_wisp, Action.RIGHT_JUMP, s_wisp, ev, False)
        self.assertAlmostEqual(r_jump_wisp, r_walk)

        # 5. Jump when laser is threatening at 50px (<= 55px threshold) -> EXEMPT
        s_laser = make_raw_state(is_grounded=True, dist_to_laser=50.0)
        r_jump_laser = strategy.calculate_reward(s_laser, Action.RIGHT_JUMP, s_laser, ev, False)
        self.assertAlmostEqual(r_jump_laser, r_walk)

        # 6. Airborne double jump (is_grounded=False) -> NEVER penalized by unnecessary_jump_penalty
        s_air = make_raw_state(is_grounded=False, can_double_jump=True)
        r_air_jump = strategy.calculate_reward(s_air, Action.RIGHT_JUMP, s_air, ev, False)
        self.assertAlmostEqual(r_air_jump, r_walk)

    def test_balanced_reward_timing_and_velocity_guidance(self) -> None:
        strategy = BalancedRewardStrategy(
            laser_danger_penalty=-3.0,
            ghost_advance_penalty=-3.0,
            monster_ground_rush_penalty=-3.0,
            monster_airborne_jump_penalty=-3.0,
            evasion_bonus=1.5,
            use_max_x=False,
        )

        # 1. Laser sweeping floor (angle in [0, pi)): rushing forward incurs laser_danger_penalty (-3.0)
        s_laser_danger = make_raw_state(dist_to_laser=40.0, laser_angle=1.57)
        ev = GameEvents(dx=4.0)
        r_fwd_danger = strategy.calculate_reward(s_laser_danger, Action.RIGHT, s_laser_danger, ev, False)
        # Expected: 4.0 - 0.1 (step) - 3.0 (laser danger) = 0.9
        self.assertAlmostEqual(r_fwd_danger, 0.9)

        # Laser pointing skyward (angle in [pi, 2pi)): safe window, no penalty
        s_laser_safe = make_raw_state(dist_to_laser=40.0, laser_angle=4.71)
        r_fwd_safe = strategy.calculate_reward(s_laser_safe, Action.RIGHT, s_laser_safe, ev, False)
        self.assertAlmostEqual(r_fwd_safe, 3.9)

        # 2. Ghost advancing over pit (ghost_vx < 0, pit ahead): jumping over incurs ghost_advance_penalty
        s_ghost_adv = make_raw_state(dist_to_ghost=50.0, ghost_vx=-1.5, dist_to_pit=40.0)
        r_jump_adv = strategy.calculate_reward(s_ghost_adv, Action.RIGHT_JUMP, s_ghost_adv, ev, False)
        # 4.0 - 0.1 - 3.0 = 0.9
        self.assertAlmostEqual(r_jump_adv, 0.9)

        # 3. Quy Nhap Trang charging on ground (vx < -2.0):
        # Staying on ground or walking into it incurs monster_ground_rush_penalty (-3.0)
        s_demon_sprint = make_raw_state(dist_to_monster=50.0, monster_vx=-4.0, monster_type="quy_nhap_trang")
        r_walk_demon = strategy.calculate_reward(s_demon_sprint, Action.RIGHT, s_demon_sprint, ev, False)
        self.assertAlmostEqual(r_walk_demon, 0.9)
        # Jumping to clear demon earns evasion_bonus (+1.5)
        r_jump_demon = strategy.calculate_reward(s_demon_sprint, Action.RIGHT_JUMP, s_demon_sprint, ev, False)
        # 4.0 - 0.1 + 1.5 = 5.4
        self.assertAlmostEqual(r_jump_demon, 5.4)

        # 4. Cuong Thi airborne hopping (vy < -1.0):
        # Jumping up into mid-air hopping vampire incurs monster_airborne_jump_penalty (-3.0)
        s_jiangshi_air = make_raw_state(dist_to_monster=50.0, monster_vy=-4.5, monster_type="cuong_thi")
        r_jump_jiangshi = strategy.calculate_reward(s_jiangshi_air, Action.RIGHT_JUMP, s_jiangshi_air, ev, False)
        self.assertAlmostEqual(r_jump_jiangshi, 0.9)

        # 5. Laser closing window (skyward angle about to cross into ground sweep via laser_rot_speed)
        s_laser_closing = make_raw_state(dist_to_laser=40.0, laser_angle=6.10, laser_rot_speed=0.05)
        r_fwd_closing = strategy.calculate_reward(s_laser_closing, Action.RIGHT, s_laser_closing, ev, False)
        self.assertAlmostEqual(r_fwd_closing, 0.9)

        # 6. Wisp low ahead (leap over) vs high overhead (do not jump into) vs charging
        s_wisp_low = make_raw_state(dist_to_wisp=30.0, wisp_dir=1, wisp_vx=-2.5, wisp_y=314.0)
        self.assertAlmostEqual(strategy.calculate_reward(s_wisp_low, Action.RIGHT, s_wisp_low, ev, False), 0.9)
        self.assertAlmostEqual(strategy.calculate_reward(s_wisp_low, Action.RIGHT_JUMP, s_wisp_low, ev, False), 5.4)

        s_wisp_high = make_raw_state(player_y=288.0, dist_to_wisp=30.0, wisp_dir=1, wisp_vx=-1.8, wisp_y=240.0)
        # 4.0 - 0.1 - 1.5 (unnecessary jump under overhead wisp) - 3.0 (wisp_danger_penalty) = -0.6
        self.assertAlmostEqual(strategy.calculate_reward(s_wisp_high, Action.RIGHT_JUMP, s_wisp_high, ev, False), -0.6)

        s_wisp_charge = make_raw_state(dist_to_wisp=-1.0, wisp_charging=True)
        # 4.0 - 0.1 - 1.5 (unnecessary jump) - 1.5 (0.5 * wisp_danger_penalty) = 0.9
        self.assertAlmostEqual(strategy.calculate_reward(s_wisp_charge, Action.RIGHT_JUMP, s_wisp_charge, ev, False), 0.9)

    def test_dense_reward_ghost_and_item(self) -> None:
        strategy = DenseRewardStrategy(death_penalty=-60.0, item_bonus=20.0)
        self.assertIsInstance(strategy, BaseRewardStrategy)

        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        r_ghost = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True)
        self.assertLess(r_ghost, 0.0)

        ev_item = GameEvents(dx=4.0, collected_item=True)
        r_item = strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False)
        self.assertGreater(r_item, 20.0)

    def test_sparse_reward_ghost_and_item(self) -> None:
        strategy = SparseRewardStrategy(goal_bonus=100.0, death_penalty=-50.0, item_bonus=10.0)
        self.assertIsInstance(strategy, BaseRewardStrategy)

        ev_ghost = GameEvents(dx=4.0, hit_ghost=True)
        self.assertEqual(strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_ghost, True), -50.0)

        ev_item = GameEvents(dx=4.0, collected_item=True)
        self.assertEqual(strategy.calculate_reward(self.s1, Action.RIGHT, self.s2, ev_item, False), 10.0)


class TestTabularQAgent(unittest.TestCase):
    def test_implements_interface(self) -> None:
        agent = TabularQAgent()
        self.assertIsInstance(agent, BaseAgent)

    def test_action_selection_and_update(self) -> None:
        agent = TabularQAgent(seed=42)
        s1 = (1, 0, 0, 0, 0, 0, 0)
        s2 = (1, 0, 1, 0, 0, 0, 0)

        action = agent.select_action(s1)
        self.assertIn(action, list(range(Action.NUM_ACTIONS)))

        # Update Q-table
        td_err = agent.update(s1, action, reward=10.0, next_state=s2, done=False)
        self.assertIsInstance(td_err, float)
        self.assertGreater(agent.q_table[s1][action], 0.0)

    def test_save_and_load_pickle_and_json(self) -> None:
        agent = TabularQAgent(seed=123)
        s = (1, 0, 2, 0, 1, 0, 0)
        agent.q_table[s][Action.RIGHT] = 42.5

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "agent.pkl")
            json_path = os.path.join(tmpdir, "agent.json")

            # Test Pickle
            agent.save(pkl_path)
            new_agent_pkl = TabularQAgent()
            new_agent_pkl.load_q_table(pkl_path)  # Test alias
            self.assertAlmostEqual(new_agent_pkl.q_table[s][Action.RIGHT], 42.5)

            # Test JSON
            agent.save(json_path)
            new_agent_json = TabularQAgent()
            new_agent_json.load(json_path)
            self.assertAlmostEqual(new_agent_json.q_table[s][Action.RIGHT], 42.5)


class TestFullSystemIntegration(unittest.TestCase):
    """End-to-end headless integration of Environment, Discretizer, Reward, and Agent."""

    def test_train_steps_all_5_levels(self) -> None:
        from env import LEVEL_BUILDERS, create_env

        discretizer = MarioStateDiscretizer()
        reward_strategy = BalancedRewardStrategy()
        agent = TabularQAgent(seed=42)

        for level_id in LEVEL_BUILDERS:
            env = create_env(level_id)
            raw_state = env.reset()
            discrete_state = discretizer.discretize(raw_state)

            for _ in range(100):
                action = agent.select_action(discrete_state)
                next_raw_state, events, done = env.step(action)
                next_discrete_state = discretizer.discretize(next_raw_state)

                reward = reward_strategy.calculate_reward(
                    raw_state, action, next_raw_state, events, done
                )
                agent.update(discrete_state, action, reward, next_discrete_state, done)

                if done:
                    raw_state = env.reset()
                    discrete_state = discretizer.discretize(raw_state)
                else:
                    raw_state = next_raw_state
                    discrete_state = next_discrete_state

        # Assert agent discovered discrete states across the 5 levels
        self.assertGreater(len(agent.q_table), 0)

    def test_trainer_saves_best_route(self) -> None:
        import json
        from env import create_env
        from training.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                env_factory=create_env,
                discretizer=MarioStateDiscretizer(),
                reward_strategy=BalancedRewardStrategy(),
                eval_interval=5,
                eval_episodes_periodic=2,
                final_eval_episodes=2,
                untrained_eval_episodes=1,
                checkpoints_dir=os.path.join(tmpdir, "checkpoints"),
                outputs_dir=os.path.join(tmpdir, "outputs"),
                agent_type="tabular",
                reward_name="balanced",
            )
            trainer.run_experiment(
                agent_factory=lambda: TabularQAgent(seed=42),
                mode="separate",
                levels=[1],
                episodes_per_level=5,
            )
            route_path = os.path.join(
                tmpdir, "checkpoints", "tabular", "separate_balanced", "best_route_lvl1.json"
            )
            self.assertTrue(os.path.exists(route_path))
            with open(route_path, "r", encoding="utf-8") as f:
                route = json.load(f)
            self.assertEqual(route["level_id"], 1)
            self.assertEqual(len(route["actions"]), route["steps"])
            self.assertEqual(len(route["positions"]), route["steps"] + 1)


if __name__ == "__main__":
    unittest.main()
