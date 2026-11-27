"""Curriculum level definitions for Hoá's Midnight Escape.

Encapsulates map layouts (obstacles, pits, ghosts, items, goal) and
stage-specific rules into immutable LevelConfig dataclasses.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from core import BaseEnvironment
from env.entities import (
    Ghost, LaserGhost, WispGhost, GoalFlag, Obstacle, Pit, WarmthItem,
    GraveMonster, QuyNhapTrang, CuongThi
)
from env.game_env import GameEnvironment


DEFAULT_FLOOR_Y: float = 320.0


@dataclass
class LevelConfig:
    """Configuration bundle defining an individual curriculum stage."""
    name: str
    obstacles: List[Obstacle]
    pits: List[Pit]
    goal: GoalFlag
    ghosts: List[Ghost]
    items: List[WarmthItem]
    monsters: Optional[List[GraveMonster]] = None
    floor_y: float = DEFAULT_FLOOR_Y
    max_steps: int = 1000
    level_id: int = 1


def _build_level_1() -> LevelConfig:
    """Level 1 (Quiet Graveyard): Introduction to basic jumping over static tombstones."""
    return LevelConfig(
        name="Level 1: Quiet Graveyard",
        obstacles=[
            Obstacle(x=320.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=640.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
        ],
        pits=[],
        ghosts=[],
        items=[],
        goal=GoalFlag(x=900.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=800,
    )


def _build_level_2() -> LevelConfig:
    """Level 2 (Open Grave Pits): Introduces lethal gap hazards requiring timed leaps."""
    return LevelConfig(
        name="Level 2: Open Graves Pits",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
            Obstacle(x=560.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0),
            Obstacle(x=880.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),  # 70px wide gap
            Pit(x_start=680.0, x_end=755.0),  # 75px wide gap
        ],
        ghosts=[],
        items=[],
        goal=GoalFlag(x=1100.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_3() -> LevelConfig:
    """Level 3 (Whispering Graves): 3 tombstones, 2 pits (including 1 hellfire), 2 uniform ghosts (jump-over & walk-under), and warmth items."""
    return LevelConfig(
        name="Level 3: Whispering Graves",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),
            Obstacle(x=560.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),
            Obstacle(x=920.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),  # 70px wide
            Pit(x_start=760.0, x_end=830.0, has_fire=True, fire_offset=0),  # 70px wide with fire
        ],
        ghosts=[
            # Ghost 1 (low -> jump OVER): default speed=1.5, float_amplitude=8.0, float_speed=0.08
            Ghost(x=640.0, y=DEFAULT_FLOOR_Y - 40.0, patrol_min_x=610.0, patrol_max_x=740.0),
            # Ghost 2 (high -> walk UNDER): same default speed=1.5, float_amplitude=8.0, float_speed=0.08
            Ghost(x=1020.0, y=DEFAULT_FLOOR_Y - 84.0, patrol_min_x=980.0, patrol_max_x=1150.0),
        ],
        items=[
            WarmthItem(x=249.0, y=DEFAULT_FLOOR_Y - 86.0),   # Above tombstone 1
            WarmthItem(x=480.0, y=DEFAULT_FLOOR_Y - 28.0),   # After pit 1
            WarmthItem(x=1050.0, y=DEFAULT_FLOOR_Y - 28.0),  # Underneath high ghost 2
        ],
        goal=GoalFlag(x=1220.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1000,
    )


def _build_level_4() -> LevelConfig:
    """Level 4 (Nightmare Catacombs): 5 tombstones, 4 hellfire pits, 5 diverse ghosts including dual counter-rotating laser scissors, fast twitcher demon, and 5 warmth items."""
    return LevelConfig(
        name="Level 4: Nightmare Catacombs",
        obstacles=[
            # Tight stone pillars and narrow landing islands between pits
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),
            Obstacle(x=620.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="buddhist"),
            Obstacle(x=980.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),       # Island between Pit 2 & 3
            Obstacle(x=1420.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="buddhist"),
            Obstacle(x=1850.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),
        ],
        pits=[
            # Pit 1: Hellmouth with active flame
            Pit(x_start=400.0, x_end=480.0, has_fire=True, fire_offset=0),
            # Pit 2 & 3: Cascading alternating fire trap with narrow island at x=980
            Pit(x_start=880.0, x_end=960.0, has_fire=True, fire_offset=25),
            Pit(x_start=1020.0, x_end=1100.0, has_fire=True, fire_offset=65),
            # Pit 4: Final wide pit gap
            Pit(x_start=1620.0, x_end=1710.0, has_fire=True, fire_offset=45),
        ],
        ghosts=[
            # 1. Low ceiling guard patrolling directly over Pit 1
            Ghost(
                x=390.0, y=DEFAULT_FLOOR_Y - 42.0,
                patrol_min_x=360.0, patrol_max_x=510.0,
                speed=1.4, float_amplitude=6.0, float_speed=0.08,
            ),
            # 2. Fast twitching speed demon
            Ghost(
                x=750.0, y=DEFAULT_FLOOR_Y - 80.0,
                patrol_min_x=680.0, patrol_max_x=840.0,
                speed=-3.0, float_amplitude=8.0, float_speed=0.18,
            ),
            # 3 & 4: Dual Counter-Rotating "Laser Scissors" (Crossfire)
            LaserGhost(
                x=1200.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1150.0, patrol_max_x=1360.0,
                speed=1.5, float_amplitude=10.0, float_speed=0.09,
                laser_length=95.0, laser_rotation_speed=0.045, laser_initial_angle=0.0,
            ),
            LaserGhost(
                x=1320.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1220.0, patrol_max_x=1400.0,
                speed=-1.5, float_amplitude=10.0, float_speed=0.09,
                laser_length=95.0, laser_rotation_speed=-0.045, laser_initial_angle=3.14,
            ),
            # 5. Final long-range sweeper guarding Khuê Văn Các
            LaserGhost(
                x=1800.0, y=DEFAULT_FLOOR_Y - 85.0,
                patrol_min_x=1740.0, patrol_max_x=1960.0,
                speed=1.8, float_amplitude=12.0, float_speed=0.11,
                laser_length=110.0, laser_rotation_speed=0.035,
            ),
        ],
        items=[
            WarmthItem(x=229.0, y=DEFAULT_FLOOR_Y - 90.0),   # Perched high on tombstone 1
            WarmthItem(x=440.0, y=DEFAULT_FLOOR_Y - 85.0),   # Mid-air above Hellmouth pit
            WarmthItem(x=990.0, y=DEFAULT_FLOOR_Y - 86.0),   # On narrow island between fire pits
            WarmthItem(x=1260.0, y=DEFAULT_FLOOR_Y - 28.0),  # Under the laser scissors
            WarmthItem(x=1750.0, y=DEFAULT_FLOOR_Y - 28.0),  # Before final sanctuary
        ],
        goal=GoalFlag(x=2050.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1400,
    )


def _build_level_5() -> LevelConfig:
    """Level 5 (Valley of the Wisp Maidens): Intermediate hard stage featuring a sky filled with Wisp Maidens near the clouds, a gauntlet of sweeping LaserGhosts along the ground, low patrolling base ghosts, and a few ground-level wisp summoners."""
    return LevelConfig(
        name="Level 5: Valley of the Wisp Maidens",
        obstacles=[
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
            Obstacle(x=580.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),
            Obstacle(x=960.0, y=DEFAULT_FLOOR_Y - 50.0, width=32.0, height=50.0, style="buddhist"),
            Obstacle(x=1320.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),
            Obstacle(x=1680.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="buddhist"),
            Obstacle(x=2040.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),
            Obstacle(x=2320.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
        ],
        pits=[
            Pit(x_start=380.0, x_end=450.0),                                       # 70px normal gap
            Pit(x_start=760.0, x_end=835.0, has_fire=True, fire_offset=0),        # 75px hellfire pit
            Pit(x_start=1140.0, x_end=1215.0),                                     # 75px normal gap
            Pit(x_start=1500.0, x_end=1580.0, has_fire=True, fire_offset=30),     # 80px hellfire pit
            Pit(x_start=1860.0, x_end=1940.0, has_fire=True, fire_offset=60),     # 80px hellfire pit
        ],
        ghosts=[
            # =========================================================================
            # 1. HIGH SKY & MID-AIR CELESTIAL SQUADRON (Wisp Maidens filling the heavens)
            # Floating near the clouds (y=70..155) raining spectral fire across the level
            # =========================================================================
            # High Sky Maiden 1: Floating right beneath the scroll clouds
            WispGhost(
                x=260.0, y=75.0,
                patrol_min_x=160.0, patrol_max_x=380.0,
                speed=1.3, float_amplitude=7.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=0, wisp_mode="high",
            ),
            # Mid-Air Maiden 2: Hovering between clouds and ground
            WispGhost(
                x=520.0, y=145.0,
                patrol_min_x=420.0, patrol_max_x=640.0,
                speed=-1.4, float_amplitude=8.0, float_speed=0.09,
                wisp_cycle=140, wisp_offset=45, wisp_mode="low",
            ),
            # High Sky Maiden 3: Soaring high above Pit 2
            WispGhost(
                x=800.0, y=80.0,
                patrol_min_x=700.0, patrol_max_x=920.0,
                speed=1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=20, wisp_mode="alternating",
            ),
            # Mid-Air Maiden 4: Central sky sentinel over Obstacle 3
            WispGhost(
                x=1080.0, y=140.0,
                patrol_min_x=980.0, patrol_max_x=1200.0,
                speed=-1.3, float_amplitude=9.0, float_speed=0.09,
                wisp_cycle=140, wisp_offset=80, wisp_mode="high",
            ),
            # High Sky Maiden 5: Cloud dweller above Pit 3
            WispGhost(
                x=1380.0, y=70.0,
                patrol_min_x=1280.0, patrol_max_x=1500.0,
                speed=1.4, float_amplitude=7.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=15, wisp_mode="low",
            ),
            # Mid-Air Maiden 6: Mid-sky artillery over the deep hellfire chasm
            WispGhost(
                x=1640.0, y=155.0,
                patrol_min_x=1540.0, patrol_max_x=1760.0,
                speed=-1.4, float_amplitude=8.0, float_speed=0.09,
                wisp_cycle=135, wisp_offset=65, wisp_mode="alternating",
            ),
            # High Sky Maiden 7: Floating near the Blood Moon and upper cloud wisps
            WispGhost(
                x=1920.0, y=85.0,
                patrol_min_x=1820.0, patrol_max_x=2040.0,
                speed=1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=125, wisp_offset=30, wisp_mode="high",
            ),
            # Mid-Air Maiden 8: High priestess defending the approach to Khuê Văn Các
            WispGhost(
                x=2200.0, y=145.0,
                patrol_min_x=2100.0, patrol_max_x=2320.0,
                speed=-1.5, float_amplitude=8.0, float_speed=0.10,
                wisp_cycle=120, wisp_offset=90, wisp_mode="alternating",
            ),

            # =========================================================================
            # 2. GROUND-LEVEL LASER SENTINELS (Sweeping 360° laser beams along the grass)
            # =========================================================================
            # Laser Sentry 1: Clockwise laser monk guarding the first crossing
            LaserGhost(
                x=460.0, y=DEFAULT_FLOOR_Y - 84.0,
                patrol_min_x=400.0, patrol_max_x=530.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.08,
                laser_length=85.0, laser_rotation_speed=0.040, laser_initial_angle=0.0,
            ),
            # Laser Sentry 2: Counter-clockwise monk sweeping over Pit 2
            LaserGhost(
                x=900.0, y=DEFAULT_FLOOR_Y - 86.0,
                patrol_min_x=840.0, patrol_max_x=960.0,
                speed=-1.3, float_amplitude=9.0, float_speed=0.09,
                laser_length=90.0, laser_rotation_speed=-0.042, laser_initial_angle=1.57,
            ),
            # Laser Sentry 3: Fast sweeping monk between Pit 3 and Obstacle 4
            LaserGhost(
                x=1240.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1180.0, patrol_max_x=1310.0,
                speed=1.5, float_amplitude=8.0, float_speed=0.08,
                laser_length=90.0, laser_rotation_speed=0.045, laser_initial_angle=3.14,
            ),
            # Laser Sentry 4: Wide reach laser monk defending Obstacle 5
            LaserGhost(
                x=1600.0, y=DEFAULT_FLOOR_Y - 85.0,
                patrol_min_x=1530.0, patrol_max_x=1660.0,
                speed=-1.4, float_amplitude=10.0, float_speed=0.09,
                laser_length=95.0, laser_rotation_speed=-0.038, laser_initial_angle=0.78,
            ),
            # Laser Sentry 5: Grand Sentinel with long laser guarding sanctuary courtyard
            LaserGhost(
                x=2020.0, y=DEFAULT_FLOOR_Y - 88.0,
                patrol_min_x=1950.0, patrol_max_x=2100.0,
                speed=1.6, float_amplitude=10.0, float_speed=0.09,
                laser_length=100.0, laser_rotation_speed=0.036, laser_initial_angle=2.35,
            ),

            # =========================================================================
            # 3. GROUND-LEVEL WISP MAIDENS (A few low summoners among the graves)
            # =========================================================================
            # Ground Maiden 1: Skimming shots rolling along the cemetery turf
            WispGhost(
                x=680.0, y=DEFAULT_FLOOR_Y - 72.0,
                patrol_min_x=610.0, patrol_max_x=740.0,
                speed=-1.3, float_amplitude=7.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=50, wisp_mode="low",
            ),
            # Ground Maiden 2: Alternating close-range fire before Pit 5
            WispGhost(
                x=1780.0, y=DEFAULT_FLOOR_Y - 74.0,
                patrol_min_x=1710.0, patrol_max_x=1840.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=25, wisp_mode="alternating",
            ),

            # =========================================================================
            # 4. GROUND-LEVEL BASE GHOSTS (Low floating sentries that force timed hops)
            # =========================================================================
            # Base Ghost 1: Right above grass before Pit 1
            Ghost(
                x=320.0, y=DEFAULT_FLOOR_Y - 40.0,
                patrol_min_x=260.0, patrol_max_x=360.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.09,
            ),
            # Base Ghost 2: Low patroller in the mid graveyard section
            Ghost(
                x=1020.0, y=DEFAULT_FLOOR_Y - 42.0,
                patrol_min_x=960.0, patrol_max_x=1060.0,
                speed=-1.4, float_amplitude=7.0, float_speed=0.08,
            ),
            # Base Ghost 3: Low patroller between the deep hellfire chasms
            Ghost(
                x=1440.0, y=DEFAULT_FLOOR_Y - 40.0,
                patrol_min_x=1380.0, patrol_max_x=1480.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.09,
            ),
        ],
        items=[
            WarmthItem(x=249.0, y=DEFAULT_FLOOR_Y - 86.0),   # Atop Tombstone 1
            WarmthItem(x=620.0, y=DEFAULT_FLOOR_Y - 28.0),   # Ground reward between hazards
            WarmthItem(x=990.0, y=DEFAULT_FLOOR_Y - 88.0),   # Perched atop Tombstone 3
            WarmthItem(x=1175.0, y=DEFAULT_FLOOR_Y - 85.0),  # Mid-air over Pit 3
            WarmthItem(x=1710.0, y=DEFAULT_FLOOR_Y - 28.0),  # Ground rest point before Zone 5
            WarmthItem(x=2240.0, y=DEFAULT_FLOOR_Y - 28.0),  # Final warmth before Khuê Văn Các
        ],
        goal=GoalFlag(x=2460.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=10000,
    )


def _build_level_6() -> LevelConfig:
    """Level 6 (Awakening of the Corpse Demon): Medium difficulty stage introducing Quỷ Nhập Tràng ambushes combined with sweeping LaserGhosts, Wisp Maidens, flame pits, and ancient monuments."""
    return LevelConfig(
        name="Level 6: Awakening of the Corpse Demon",
        obstacles=[
            Obstacle(x=220.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
            Obstacle(x=520.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),       # Ambush Grave 1
            Obstacle(x=860.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="buddhist"),
            Obstacle(x=1220.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),     # Ambush Grave 2
            Obstacle(x=1540.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
        ],
        pits=[
            Pit(x_start=360.0, x_end=430.0),                                       # 70px normal pit
            Pit(x_start=700.0, x_end=775.0, has_fire=True, fire_offset=0),         # 75px hellfire pit
            Pit(x_start=1060.0, x_end=1130.0),                                     # 70px gap for Quỷ Nhập Tràng pit-vaulting
        ],
        ghosts=[
            # 1. Base hovering ghost (low float before Pit 1)
            Ghost(
                x=300.0, y=DEFAULT_FLOOR_Y - 40.0,
                patrol_min_x=260.0, patrol_max_x=340.0,
                speed=1.4, float_amplitude=7.0, float_speed=0.08,
            ),
            # 2. Wisp Maiden casting rolling will-o'-the-wisps across the graveyard turf
            WispGhost(
                x=620.0, y=DEFAULT_FLOOR_Y - 76.0,
                patrol_min_x=570.0, patrol_max_x=680.0,
                speed=-1.3, float_amplitude=8.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=30, wisp_mode="low",
            ),
            # 3. Rotating LaserGhost sentry guarding the middle terrace
            LaserGhost(
                x=960.0, y=DEFAULT_FLOOR_Y - 84.0,
                patrol_min_x=900.0, patrol_max_x=1030.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.08,
                laser_length=85.0, laser_rotation_speed=0.038, laser_initial_angle=0.0,
            ),
            # 4. High hovering ghost before the sanctuary (walk/slide under)
            Ghost(
                x=1420.0, y=DEFAULT_FLOOR_Y - 86.0,
                patrol_min_x=1360.0, patrol_max_x=1490.0,
                speed=-1.3, float_amplitude=8.0, float_speed=0.08,
            ),
        ],
        monsters=[
            # Corpse Demon 1: Ambushes from the cross tombstone at x=520, accelerates into sprint
            QuyNhapTrang(
                x=520.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=170.0, emerge_speed=1.6,
                base_speed=1.4, max_speed=4.8, acceleration=0.04,
            ),
            # Corpse Demon 2: Ambushes from tombstone at x=1220, vaults over Pit 3 across to Hoá!
            QuyNhapTrang(
                x=1220.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=180.0, emerge_speed=1.8,
                base_speed=1.6, max_speed=5.2, acceleration=0.05,
            ),
        ],
        items=[
            WarmthItem(x=229.0, y=DEFAULT_FLOOR_Y - 86.0),   # Atop Tombstone 1
            WarmthItem(x=395.0, y=DEFAULT_FLOOR_Y - 85.0),   # Airborne over Pit 1
            WarmthItem(x=650.0, y=DEFAULT_FLOOR_Y - 28.0),   # Rewarding ground item between hazards
            WarmthItem(x=1095.0, y=DEFAULT_FLOOR_Y - 85.0),  # Airborne over Pit 3
            WarmthItem(x=1430.0, y=DEFAULT_FLOOR_Y - 28.0),  # Beneath high ghost 4
            WarmthItem(x=1620.0, y=DEFAULT_FLOOR_Y - 28.0),  # Final garlic before sanctuary
        ],
        goal=GoalFlag(x=1720.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=1500,
    )


def _build_level_7() -> LevelConfig:
    """Level 7 (Cataclysm of the Undead): Master challenge stage combining all entities.
    Features Jiangshi (Cương Thi) and Corpse Demon (Quỷ Nhập Tràng) ambushes (each tied to a separate tombstone),
    staggered hellfire chasms, floating stone platforms, counter-rotating laser scissors, and wisp maidens.
    """
    return LevelConfig(
        name="Level 7: Cataclysm of the Undead",
        obstacles=[
            # 1. Buddhist stele before first pit
            Obstacle(x=240.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),
            # 2. Granite cross tombstone - Ambush site for Jiangshi 1
            Obstacle(x=520.0, y=DEFAULT_FLOOR_Y - 58.0, width=32.0, height=58.0, style="cross"),
            # 3. Buddhist pagoda tombstone - Ambush site for Corpse Demon 1
            Obstacle(x=960.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="buddhist"),
            # 4. Floating stone platform over hellfire zone
            Obstacle(x=1270.0, y=DEFAULT_FLOOR_Y - 72.0, width=48.0, height=16.0, style="platform"),
            # 5. Grand Catholic cross - Ambush site for Jiangshi 2
            Obstacle(x=1680.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="cross"),
            # 6. Floating stone platform over second hellfire zone
            Obstacle(x=1980.0, y=DEFAULT_FLOOR_Y - 74.0, width=48.0, height=16.0, style="platform"),
            # 7. Buddhist pagoda tombstone - Ambush site for Corpse Demon 2
            Obstacle(x=2340.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),
            # 8. Grand granite monument guarding final stretch
            Obstacle(x=2720.0, y=DEFAULT_FLOOR_Y - 68.0, width=32.0, height=68.0, style="cross"),
        ],
        pits=[
            Pit(x_start=370.0, x_end=440.0),                                       # 70px normal pit
            Pit(x_start=730.0, x_end=810.0, has_fire=True, fire_offset=0),         # 80px hellfire pit
            Pit(x_start=1140.0, x_end=1220.0, has_fire=True, fire_offset=75),     # 80px hellfire pit (opposite phase)
            Pit(x_start=1440.0, x_end=1510.0),                                     # 70px pit under laser scissors
            Pit(x_start=1810.0, x_end=1895.0, has_fire=True, fire_offset=30),     # 85px hellfire chasm
            Pit(x_start=2470.0, x_end=2550.0, has_fire=True, fire_offset=90),     # 80px hellfire pit before sanctuary
        ],
        ghosts=[
            # 1. Low skimming ground ghost before Pit 1
            Ghost(
                x=310.0, y=DEFAULT_FLOOR_Y - 40.0,
                patrol_min_x=260.0, patrol_max_x=360.0,
                speed=1.5, float_amplitude=7.0, float_speed=0.09,
            ),
            # 2. Sweeping clockwise LaserGhost
            LaserGhost(
                x=660.0, y=DEFAULT_FLOOR_Y - 110.0,
                patrol_min_x=610.0, patrol_max_x=710.0,
                speed=-1.2, float_amplitude=6.0, float_speed=0.06,
                laser_length=85.0, laser_rotation_speed=0.038, laser_initial_angle=0.0,
            ),
            # 3. High Sky Wisp Maiden over Hellfire Pit 2
            WispGhost(
                x=770.0, y=DEFAULT_FLOOR_Y - 130.0,
                patrol_min_x=730.0, patrol_max_x=820.0,
                speed=1.3, float_amplitude=8.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=0, wisp_mode="high",
            ),
            # 4. Low Ground Wisp Maiden casting rolling turf flames
            WispGhost(
                x=1050.0, y=DEFAULT_FLOOR_Y - 72.0,
                patrol_min_x=990.0, patrol_max_x=1110.0,
                speed=-1.3, float_amplitude=7.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=40, wisp_mode="low",
            ),
            # 5. Counter-rotating Laser Scissors 1 (Clockwise)
            LaserGhost(
                x=1380.0, y=DEFAULT_FLOOR_Y - 115.0,
                patrol_min_x=1340.0, patrol_max_x=1420.0,
                speed=1.1, float_amplitude=6.0, float_speed=0.06,
                laser_length=85.0, laser_rotation_speed=0.035, laser_initial_angle=0.0,
            ),
            # 6. Counter-rotating Laser Scissors 2 (Counter-Clockwise)
            LaserGhost(
                x=1500.0, y=DEFAULT_FLOOR_Y - 115.0,
                patrol_min_x=1460.0, patrol_max_x=1540.0,
                speed=-1.1, float_amplitude=6.0, float_speed=0.06,
                laser_length=85.0, laser_rotation_speed=-0.035, laser_initial_angle=3.14,
            ),
            # 7. Alternating Wisp Maiden between chasms
            WispGhost(
                x=1760.0, y=DEFAULT_FLOOR_Y - 95.0,
                patrol_min_x=1710.0, patrol_max_x=1810.0,
                speed=1.4, float_amplitude=8.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=25, wisp_mode="alternating",
            ),
            # 8. Fast twitcher demon ghost over grass
            Ghost(
                x=2140.0, y=DEFAULT_FLOOR_Y - 42.0,
                patrol_min_x=2070.0, patrol_max_x=2210.0,
                speed=-1.7, float_amplitude=9.0, float_speed=0.10,
            ),
            # 9. Clockwise LaserGhost guarding the final pit
            LaserGhost(
                x=2510.0, y=DEFAULT_FLOOR_Y - 118.0,
                patrol_min_x=2470.0, patrol_max_x=2560.0,
                speed=1.2, float_amplitude=7.0, float_speed=0.07,
                laser_length=90.0, laser_rotation_speed=0.04, laser_initial_angle=1.57,
            ),
            # 10. Sky Wisp Maiden raining fire
            WispGhost(
                x=2620.0, y=DEFAULT_FLOOR_Y - 130.0,
                patrol_min_x=2580.0, patrol_max_x=2670.0,
                speed=-1.3, float_amplitude=8.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=60, wisp_mode="high",
            ),
            # 11. High hovering sentry before sanctuary
            Ghost(
                x=2820.0, y=DEFAULT_FLOOR_Y - 86.0,
                patrol_min_x=2760.0, patrol_max_x=2880.0,
                speed=-1.3, float_amplitude=8.0, float_speed=0.08,
            ),
        ],
        monsters=[
            # Monster 1: Jiangshi (Cương Thi) ambushes from Cross Tombstone at x=520
            CuongThi(
                x=520.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=190.0, hop_speed_x=2.5,
                hop_force_y=-5.6, pause_duration=16,
            ),
            # Monster 2: Corpse Demon (Quỷ Nhập Tràng) ambushes from Buddhist Tombstone at x=960, vaults Pit 3
            QuyNhapTrang(
                x=960.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=180.0, emerge_speed=1.8,
                base_speed=1.5, max_speed=5.0, acceleration=0.045,
            ),
            # Monster 3: Agile Jiangshi (Cương Thi) ambushes from Cross Tombstone at x=1680, vaults Pit 4
            CuongThi(
                x=1680.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=200.0, hop_speed_x=2.8,
                hop_force_y=-5.8, pause_duration=13,
            ),
            # Monster 4: Berserk Corpse Demon (Quỷ Nhập Tràng) ambushes from Tombstone at x=2340, vaults Pit 6
            QuyNhapTrang(
                x=2340.0, y=DEFAULT_FLOOR_Y - 32.0,
                trigger_dist=190.0, emerge_speed=1.9,
                base_speed=1.6, max_speed=5.4, acceleration=0.05,
            ),
        ],
        items=[
            WarmthItem(x=249.0, y=DEFAULT_FLOOR_Y - 86.0),   # Atop Tombstone 1
            WarmthItem(x=405.0, y=DEFAULT_FLOOR_Y - 85.0),   # Airborne over Pit 1
            WarmthItem(x=650.0, y=DEFAULT_FLOOR_Y - 28.0),   # Ground reward before Hellfire Pit 2
            WarmthItem(x=969.0, y=DEFAULT_FLOOR_Y - 88.0),   # Atop Tombstone 3
            WarmthItem(x=1290.0, y=DEFAULT_FLOOR_Y - 110.0), # Perched atop Floating Stone Platform 1
            WarmthItem(x=1850.0, y=DEFAULT_FLOOR_Y - 88.0),  # Airborne over Hellfire Chasm 5
            WarmthItem(x=2000.0, y=DEFAULT_FLOOR_Y - 112.0), # Perched atop Floating Stone Platform 2
            WarmthItem(x=2730.0, y=DEFAULT_FLOOR_Y - 105.0), # Atop Monument 8 before sanctuary
        ],
        goal=GoalFlag(x=2950.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=10000,
    )

def _build_level_8() -> LevelConfig:
    """
    Level 8 (The Celestial Purgatory): A multi-layer 7,200px marathon gauntlet (~3 minutes).
    - Unified pagoda lantern stone pillars (Image 1) firmly planted on ground when ground exists,
      floating only when suspended over abysses.
    - Hanging slide tunnels with exquisite 'Lưỡng Long Chầu Sen' carved relief panels (Image 3).
    - Slide tunnel mouth guards (standard Ghosts) challenging player to slide underneath them!
    - Rich multi-tier celestial ghost sky: High sky (y~50-85), Mid sky (y~130-175), and Low sky (y~215-255).
    - Colossal voids/abysses beneath all aerial platform zones (pure void, no ground).
    - Diverse field of tombstones with empty/dummy graves mixed with ambush graves for suspense!
    - Accurate death messages distinguishing CuongThi (Jiangshi) from QuyNhapTrang (Corpse Demon).
    """
    return LevelConfig(
        name="Level 8: The Celestial Purgatory",
        obstacles=[
            # =========================================================================
            # ZONE 1: THE GATES OF PURGATORY (x = 200 -> 1,350)
            # Solid ground: all pillars & stepped altars sit firmly on ground (bottom=320.0)
            # All columns unified with the pagoda stone pillar style (Image 1)!
            # Tuned strictly for regular player's jump (JUMP_FORCE=-13.0, WALK_SPEED=4.0)
            # =========================================================================
            # Ascending Stepped Pagoda Pillars & Summit Pillar (firmly on ground, bottom=320.0)
            Obstacle(x=200.0, y=DEFAULT_FLOOR_Y - 36.0, width=38.0, height=36.0, style="pillar"),
            Obstacle(x=248.0, y=DEFAULT_FLOOR_Y - 72.0, width=38.0, height=72.0, style="pillar"),
            Obstacle(x=296.0, y=DEFAULT_FLOOR_Y - 108.0, width=40.0, height=108.0, style="pillar"),   # Summit Pillar 1 (top at y=212)
            # Descending Pagoda Pillars across Pit 1 (89px gap from x=336 to 425, reachable by normal jump!)
            Obstacle(x=425.0, y=DEFAULT_FLOOR_Y - 96.0, width=40.0, height=96.0, style="pillar"),     # Descending Landing Pillar (top at y=224)
            Obstacle(x=475.0, y=DEFAULT_FLOOR_Y - 64.0, width=38.0, height=64.0, style="pillar"),
            Obstacle(x=523.0, y=DEFAULT_FLOOR_Y - 32.0, width=38.0, height=32.0, style="pillar"),

            # Zone 1 Graves (Empty Dummy Graves + Ambush Graves for Suspense!)
            Obstacle(x=590.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),     # Empty Grave 1
            Obstacle(x=680.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),  # Ambush Grave 1 (CuongThi)
            Obstacle(x=830.0, y=DEFAULT_FLOOR_Y - 50.0, width=32.0, height=50.0, style="buddhist"),  # Empty Grave 2
            Obstacle(x=880.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),     # Ambush Grave 2 (QuyNhapTrang)
            
            # --- TUNNEL 1 (Overhead Hanging Stone Tablet with 'Lưỡng Long Chầu Sen' Relief) ---
            # Runway: x=912 -> 1080 (168px flat runway). Standard Ghost mouth guard guards entrance!
            Obstacle(x=1080.0, y=0.0, width=130.0, height=DEFAULT_FLOOR_Y - 24.0, style="hanging"),
            # Recovery ground after Tunnel 1: x=1210 -> 1340 (130px flat ground)

            # =========================================================================
            # ZONE 2: THE CELESTIAL ABYSS & FLOATING PILLARS (x = 1,340 -> 2,260)
            # Colossal bottomless void pit (pure void, no ground below).
            # All aerial platform gaps <= 60px and heights <= 50-70px so regular Mario
            # can clear every jump with normal jump & double jump (NOT tuned for God mode)!
            # =========================================================================
            # Mid-layer Floating Pagoda Pillar (Image 1, suspended over abyss, top at y=250)
            Obstacle(x=1370.0, y=DEFAULT_FLOOR_Y - 70.0, width=42.0, height=110.0, style="pillar"),
            
            # MID-LAYER SENTRY CATWALK 1 (140px wide carved lotus altar, y_top = 200.0)
            Obstacle(x=1470.0, y=DEFAULT_FLOOR_Y - 120.0, width=140.0, height=30.0, style="platform"),
            # Sentry Grave marker on Platform 1 with clear landing zones:
            Obstacle(x=1530.0, y=DEFAULT_FLOOR_Y - 120.0 - 48.0, width=32.0, height=48.0, style="cross"), # Ambush (CuongThi sentry)

            # Towering Pagoda Pillar soaring high into the sky (Image 1, top at y=150)
            Obstacle(x=1670.0, y=DEFAULT_FLOOR_Y - 170.0, width=42.0, height=210.0, style="pillar"),

            # HIGH SKY CELESTIAL CATWALK (140px wide sky platform near clouds, y_top = 100.0)
            Obstacle(x=1770.0, y=DEFAULT_FLOOR_Y - 220.0, width=140.0, height=26.0, style="platform"),
            # Sentry Grave marker sits flush ON TOP of High Sky Catwalk:
            Obstacle(x=1830.0, y=DEFAULT_FLOOR_Y - 220.0 - 48.0, width=32.0, height=48.0, style="buddhist"), # Ambush (QuyNhapTrang sentry)

            # Floating Pagoda Pillar (Image 1, top at y=150)
            Obstacle(x=1970.0, y=DEFAULT_FLOOR_Y - 170.0, width=42.0, height=210.0, style="pillar"),
            # Stepping Pagoda Pillars (Clean top surfaces for reliable player landing!)
            Obstacle(x=2070.0, y=DEFAULT_FLOOR_Y - 120.0, width=42.0, height=160.0, style="pillar"), # Top at y=200
            Obstacle(x=2170.0, y=DEFAULT_FLOOR_Y - 70.0, width=42.0, height=110.0, style="pillar"),  # Top at y=250

            # =========================================================================
            # ZONE 3: MONUMENTAL TEMPLE PYRAMID & CATACOMB SLIDE (x = 2,260 -> 3,600)
            # Solid ground returns at x=2260! All pillars sit firmly on ground (bottom=320.0).
            # All pyramid columns unified with the pagoda stone pillar style!
            # =========================================================================
            # Grand Stepped Temple Pyramid (Unified Pagoda Pillars firmly on ground)
            Obstacle(x=2300.0, y=DEFAULT_FLOOR_Y - 36.0, width=38.0, height=36.0, style="pillar"),
            Obstacle(x=2348.0, y=DEFAULT_FLOOR_Y - 72.0, width=38.0, height=72.0, style="pillar"),
            Obstacle(x=2396.0, y=DEFAULT_FLOOR_Y - 108.0, width=38.0, height=108.0, style="pillar"),
            Obstacle(x=2444.0, y=DEFAULT_FLOOR_Y - 144.0, width=54.0, height=144.0, style="pillar"), # Grand Temple Apex (y=176)
            Obstacle(x=2508.0, y=DEFAULT_FLOOR_Y - 96.0, width=38.0, height=96.0, style="pillar"),
            Obstacle(x=2556.0, y=DEFAULT_FLOOR_Y - 48.0, width=38.0, height=48.0, style="pillar"),

            # --- TUNNEL 2 (Overhead Hanging Stone Tablet with 'Lưỡng Long Chầu Sen' Relief) ---
            # 146px flat uninterrupted runway: x=2594 -> 2740. Standard Ghost mouth guard guards entrance!
            Obstacle(x=2740.0, y=0.0, width=140.0, height=DEFAULT_FLOOR_Y - 24.0, style="hanging"),
            # Flat recovery ground after Tunnel 2: x=2880 -> 3040 (160px flat ground)
            Obstacle(x=3040.0, y=DEFAULT_FLOOR_Y - 54.0, width=32.0, height=54.0, style="buddhist"),  # Ambush Grave 3 (CuongThi)
            Obstacle(x=3080.0, y=DEFAULT_FLOOR_Y - 48.0, width=30.0, height=48.0, style="cross"),     # Empty Grave 5
            Obstacle(x=3260.0, y=DEFAULT_FLOOR_Y - 62.0, width=32.0, height=62.0, style="cross"),     # Ambush Grave 4 (QuyNhapTrang)
            Obstacle(x=3300.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="buddhist"),  # Empty Grave 6
            Obstacle(x=3435.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),     # Empty Grave 7
            Obstacle(x=3480.0, y=DEFAULT_FLOOR_Y - 54.0, width=32.0, height=54.0, style="buddhist"),  # Ambush Grave 5 (CuongThi)
            Obstacle(x=3550.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="cross"),     # Empty Grave 8
            Obstacle(x=3620.0, y=DEFAULT_FLOOR_Y - 50.0, width=32.0, height=50.0, style="buddhist"),  # Empty Grave 9

            # =========================================================================
            # ZONE 4: THE VALLEY OF STONE PILLARS & CHASMS (x = 3,600 -> 5,150)
            # Solid ground: pillars firmly on ground (bottom=320.0)
            # =========================================================================
            Obstacle(x=3720.0, y=DEFAULT_FLOOR_Y - 140.0, width=42.0, height=140.0, style="pillar"),  # Valley Pillar (firmly on ground)
            Obstacle(x=3800.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),     # Empty Grave 10
            Obstacle(x=3980.0, y=DEFAULT_FLOOR_Y - 56.0, width=32.0, height=56.0, style="buddhist"),  # Ambush Grave 6 (QuyNhapTrang)
            Obstacle(x=4040.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="cross"),     # Empty Grave 11
            Obstacle(x=4100.0, y=DEFAULT_FLOOR_Y - 100.0, width=54.0, height=24.0, style="platform"), # Stepping altar (y=220)
            Obstacle(x=4270.0, y=DEFAULT_FLOOR_Y - 64.0, width=32.0, height=64.0, style="cross"),     # Ambush Grave 7 (CuongThi)
            
            # --- TUNNEL 3 (Overhead Hanging Stone Tablet with 'Lưỡng Long Chầu Sen' Relief) ---
            # 178px flat runway: x=4302 -> 4480. Standard Ghost mouth guard guards entrance!
            Obstacle(x=4480.0, y=0.0, width=140.0, height=DEFAULT_FLOOR_Y - 24.0, style="hanging"),
            # Recovery ground after Tunnel 3: x=4620 -> 4760 (140px flat runway!)
            Obstacle(x=4680.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),     # Empty Grave 12
            Obstacle(x=4760.0, y=DEFAULT_FLOOR_Y - 54.0, width=32.0, height=54.0, style="buddhist"),  # Ambush Grave 8 (QuyNhapTrang)
            Obstacle(x=4810.0, y=DEFAULT_FLOOR_Y - 48.0, width=32.0, height=48.0, style="buddhist"),  # Empty Grave 13
            Obstacle(x=4980.0, y=DEFAULT_FLOOR_Y - 60.0, width=32.0, height=60.0, style="cross"),     # Empty Grave 14
            Obstacle(x=5040.0, y=DEFAULT_FLOOR_Y - 54.0, width=32.0, height=54.0, style="buddhist"),  # Empty Grave 15

            # =========================================================================
            # ZONE 5: THE SKY CITADEL & COLOSSAL VOID (x = 5,150 -> 5,930)
            # Colossal bottomless void pit (pure void, no ground below).
            # All platform gaps <= 60px so normal Mario can clear every jump!
            # =========================================================================
            Obstacle(x=5180.0, y=DEFAULT_FLOOR_Y - 70.0, width=42.0, height=110.0, style="pillar"), # Floating Pillar 1 (top at y=250)
            Obstacle(x=5280.0, y=DEFAULT_FLOOR_Y - 130.0, width=120.0, height=24.0, style="platform"), # Floating Altar (top at y=190)
            Obstacle(x=5340.0, y=DEFAULT_FLOOR_Y - 130.0 - 45.0, width=32.0, height=45.0, style="buddhist"), # Empty Grave 16 (Mid-air)
            Obstacle(x=5460.0, y=DEFAULT_FLOOR_Y - 190.0, width=42.0, height=230.0, style="pillar"), # Towering Pillar 2 (top at y=130)
            
            # SKY CITADEL SENTRY BRIDGE (140px wide stone bridge at y_top = 80.0)
            Obstacle(x=5560.0, y=DEFAULT_FLOOR_Y - 240.0, width=140.0, height=24.0, style="platform"),
            # Sentry Grave marker sits flush ON TOP of Sky Citadel Bridge:
            Obstacle(x=5620.0, y=DEFAULT_FLOOR_Y - 240.0 - 48.0, width=32.0, height=48.0, style="cross"), # Ambush (QuyNhapTrang)

            # Floating Pagoda Pillar (Image 1, top at y=160)
            Obstacle(x=5760.0, y=DEFAULT_FLOOR_Y - 160.0, width=42.0, height=200.0, style="pillar"),
            Obstacle(x=5850.0, y=DEFAULT_FLOOR_Y - 80.0, width=42.0, height=120.0, style="pillar"),   # Floating Pillar (top at y=240)

            # --- TUNNEL 4 (Overhead Hanging Stone Tablet with 'Lưỡng Long Chầu Sen' Relief) ---
            # Solid ground returns at x=5930! 150px flat runway: x=5930 -> 6080. Standard Ghost mouth guard!
            Obstacle(x=6080.0, y=0.0, width=130.0, height=DEFAULT_FLOOR_Y - 24.0, style="hanging"),
            # Recovery ground after Tunnel 4: x=6210 -> 6330 (120px flat runway!)
            Obstacle(x=6240.0, y=DEFAULT_FLOOR_Y - 58.0, width=32.0, height=58.0, style="buddhist"),  # Ambush Grave 9 (CuongThi)
            Obstacle(x=6280.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),     # Empty Grave 18

            # =========================================================================
            # ZONE 6: CELESTIAL APEX & SACRED SANCTUARY (x = 6,330 -> 7,200)
            # Solid ground: all pillars firmly on ground (bottom=320.0)
            # All pyramid columns unified with the pagoda stone pillar style!
            # =========================================================================
            # Final Grand Celestial Pyramid Steps & Pagoda Summit Pillar (firmly on ground)
            Obstacle(x=6330.0, y=DEFAULT_FLOOR_Y - 36.0, width=38.0, height=36.0, style="pillar"),
            Obstacle(x=6378.0, y=DEFAULT_FLOOR_Y - 72.0, width=38.0, height=72.0, style="pillar"),
            Obstacle(x=6426.0, y=DEFAULT_FLOOR_Y - 108.0, width=38.0, height=108.0, style="pillar"),  # Pyramid Pillar (firmly on ground)
            Obstacle(x=6474.0, y=DEFAULT_FLOOR_Y - 150.0, width=54.0, height=150.0, style="pillar"), # Final Summit Altar (y=170)
            Obstacle(x=6538.0, y=DEFAULT_FLOOR_Y - 90.0, width=38.0, height=90.0, style="pillar"),
            Obstacle(x=6580.0, y=DEFAULT_FLOOR_Y - 50.0, width=32.0, height=50.0, style="buddhist"),  # Empty Grave 19

            # Sacred Sanctuary Gateway (Pillars lowered & Standing Tablet Monument firmly on ground for player to jump over!)
            Obstacle(x=6720.0, y=DEFAULT_FLOOR_Y - 120.0, width=42.0, height=120.0, style="pillar"),  # Firmly on ground (bottom=320)
            Obstacle(x=6750.0, y=DEFAULT_FLOOR_Y - 52.0, width=32.0, height=52.0, style="cross"),     # Empty Grave 20
            # Pit 8 at x=6800..6880, followed by clean 60px runway at x=6880 -> 6940 for jumping over the Gateway:
            Obstacle(x=6940.0, y=DEFAULT_FLOOR_Y - 56.0, width=40.0, height=56.0, style="pillar"),   # Left Gate Pillar (h=56, top at y=264)
            Obstacle(x=6990.0, y=DEFAULT_FLOOR_Y - 76.0, width=80.0, height=76.0, style="tablet"),   # Standing Tablet Monument (h=76, top at y=244)
            Obstacle(x=7080.0, y=DEFAULT_FLOOR_Y - 56.0, width=40.0, height=56.0, style="pillar"),   # Right Gate Pillar (h=56, top at y=264)
        ],
        pits=[
            # Zone 1 Chasms (85px and 80px gaps, cleanly jumpable by normal Mario!)
            Pit(x_start=340.0, x_end=425.0, has_fire=True, fire_offset=0),
            Pit(x_start=730.0, x_end=810.0, has_fire=True, fire_offset=40),

            # Zone 2: COLOSSAL HELL SEA ABYSS (Continuous 920px void under the aerial pillars & catwalks!)
            Pit(x_start=1340.0, x_end=2260.0, has_fire=False),

            # Zone 3 Chasms (80px pits)
            Pit(x_start=3120.0, x_end=3200.0, has_fire=True, fire_offset=50),
            Pit(x_start=3340.0, x_end=3420.0, has_fire=True, fire_offset=25),

            # Zone 4 Chasms (80px pits)
            Pit(x_start=3860.0, x_end=3940.0, has_fire=True, fire_offset=80),
            Pit(x_start=4170.0, x_end=4250.0, has_fire=True, fire_offset=15),
            Pit(x_start=4860.0, x_end=4940.0, has_fire=True, fire_offset=65),

            # Zone 5: COLOSSAL SKY CITADEL ABYSS (Continuous 780px void under the sky citadel catwalks!)
            Pit(x_start=5150.0, x_end=5930.0, has_fire=False),

            # Zone 6 Final Abyss (80px pits)
            Pit(x_start=6600.0, x_end=6680.0, has_fire=True, fire_offset=40),
            Pit(x_start=6800.0, x_end=6880.0, has_fire=True, fire_offset=80),
        ],
        monsters=[
            # --- Zone 1 Ground Ambushes ---
            CuongThi(
                x=680.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=170.0,
                hop_speed_x=2.4, hop_force_y=-5.4, pause_duration=16,
            ),
            QuyNhapTrang(
                x=880.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=160.0,
                base_speed=1.5, max_speed=4.8, acceleration=0.04,
            ),

            # --- Zone 2 Platform Sentries (Pacing on elevated catwalks with zero overlap & auto turnaround!) ---
            CuongThi(
                x=1530.0, y=DEFAULT_FLOOR_Y - 120.0, trigger_dist=85.0,
                platform_y=DEFAULT_FLOOR_Y - 120.0,
                patrol_min_x=1470.0, patrol_max_x=1610.0, # 140px catwalk
                hop_speed_x=2.2, hop_force_y=-5.2, pause_duration=15,
            ),
            QuyNhapTrang(
                x=1830.0, y=DEFAULT_FLOOR_Y - 220.0, trigger_dist=85.0,
                platform_y=DEFAULT_FLOOR_Y - 220.0,
                patrol_min_x=1770.0, patrol_max_x=1910.0, # High sky catwalk (y=100)
                base_speed=1.8,
            ),

            # --- Zone 3 & 4 Turf Ambushes ---
            CuongThi(
                x=3040.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=170.0,
                hop_speed_x=2.5, hop_force_y=-5.5, pause_duration=14,
            ),
            QuyNhapTrang(
                x=3260.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=170.0,
                base_speed=1.6, max_speed=5.0, acceleration=0.045,
            ),
            CuongThi(
                x=3480.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=180.0,
                hop_speed_x=2.6, hop_force_y=-5.6, pause_duration=13,
            ),
            QuyNhapTrang(
                x=3980.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=175.0,
                base_speed=1.6, max_speed=5.2, acceleration=0.05,
            ),
            CuongThi(
                x=4270.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=185.0,
                hop_speed_x=2.7, hop_force_y=-5.7, pause_duration=12,
            ),
            QuyNhapTrang(
                x=4760.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=180.0,
                base_speed=1.7, max_speed=5.3, acceleration=0.05,
            ),

            # --- Zone 5 Sky Platform Sentry ---
            QuyNhapTrang(
                x=5620.0, y=DEFAULT_FLOOR_Y - 240.0, trigger_dist=85.0,
                platform_y=DEFAULT_FLOOR_Y - 240.0,
                patrol_min_x=5560.0, patrol_max_x=5700.0, # Sky Citadel catwalk (y=80)
                base_speed=2.0,
            ),
            CuongThi(
                x=6240.0, y=DEFAULT_FLOOR_Y - 32.0, trigger_dist=170.0,
                hop_speed_x=2.8, hop_force_y=-5.8, pause_duration=12,
            ),
        ],
        ghosts=[
            # =========================================================================
            # LAYER 1: HIGH SKY TIER (y = 28 -> 55) - Upper Cloud Wisps & Sweeping Lasers
            # High above player jump arcs, providing magnificent celestial searchlights!
            # =========================================================================
            WispGhost(
                x=250.0, y=48.0, patrol_min_x=160.0, patrol_max_x=340.0,
                speed=1.2, float_amplitude=5.0, float_speed=0.07,
                wisp_cycle=130, wisp_offset=0, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 1 (over Zone 1)
            LaserGhost(
                x=650.0, y=50.0, patrol_min_x=580.0, patrol_max_x=720.0,
                speed=-1.1, float_amplitude=5.0, float_speed=0.06,
                laser_length=80.0, laser_rotation_speed=0.035,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=1200.0, y=48.0, patrol_min_x=1120.0, patrol_max_x=1280.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=20, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 2 (over Zone 2 Abyss)
            LaserGhost(
                x=1550.0, y=45.0, patrol_min_x=1480.0, patrol_max_x=1620.0,
                speed=-1.2, float_amplitude=5.0, float_speed=0.07,
                laser_length=75.0, laser_rotation_speed=0.038,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=1720.0, y=30.0, patrol_min_x=1650.0, patrol_max_x=1750.0,
                speed=1.3, float_amplitude=4.0, float_speed=0.07,
                wisp_cycle=135, wisp_offset=15, wisp_mode="high",
            ),
            WispGhost(
                x=2200.0, y=48.0, patrol_min_x=2120.0, patrol_max_x=2280.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=60, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 3 (over Zone 3 Pyramid)
            LaserGhost(
                x=2600.0, y=50.0, patrol_min_x=2520.0, patrol_max_x=2680.0,
                speed=1.1, float_amplitude=5.0, float_speed=0.06,
                laser_length=80.0, laser_rotation_speed=0.036,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=2920.0, y=48.0, patrol_min_x=2840.0, patrol_max_x=3000.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=80, wisp_mode="high",
            ),
            WispGhost(
                x=3350.0, y=50.0, patrol_min_x=3260.0, patrol_max_x=3440.0,
                speed=1.4, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=10, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 4 (over Zone 4 Valley)
            LaserGhost(
                x=3750.0, y=50.0, patrol_min_x=3670.0, patrol_max_x=3830.0,
                speed=-1.2, float_amplitude=5.0, float_speed=0.07,
                laser_length=80.0, laser_rotation_speed=0.037,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=4200.0, y=48.0, patrol_min_x=4120.0, patrol_max_x=4280.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=70, wisp_mode="high",
            ),
            WispGhost(
                x=4550.0, y=48.0, patrol_min_x=4460.0, patrol_max_x=4640.0,
                speed=-1.4, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=25, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 5 (over Zone 5 Sky Citadel)
            LaserGhost(
                x=5050.0, y=50.0, patrol_min_x=4970.0, patrol_max_x=5130.0,
                speed=1.1, float_amplitude=5.0, float_speed=0.06,
                laser_length=80.0, laser_rotation_speed=0.036,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=5350.0, y=45.0, patrol_min_x=5260.0, patrol_max_x=5440.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=35, wisp_mode="high",
            ),
            WispGhost(
                x=5500.0, y=28.0, patrol_min_x=5440.0, patrol_max_x=5540.0,
                speed=1.3, float_amplitude=4.0, float_speed=0.07,
                wisp_cycle=130, wisp_offset=85, wisp_mode="high",
            ),
            # High Sky Sweeping LaserGhost 6 (over Zone 6 Apex Approach)
            LaserGhost(
                x=6150.0, y=50.0, patrol_min_x=6070.0, patrol_max_x=6230.0,
                speed=-1.2, float_amplitude=5.0, float_speed=0.07,
                laser_length=80.0, laser_rotation_speed=0.038,
                laser_initial_angle=0.0, laser_sweep_amplitude=0.45,
            ),
            WispGhost(
                x=6500.0, y=48.0, patrol_min_x=6410.0, patrol_max_x=6590.0,
                speed=1.4, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=65, wisp_mode="high",
            ),
            WispGhost(
                x=6820.0, y=48.0, patrol_min_x=6740.0, patrol_max_x=6900.0,
                speed=-1.4, float_amplitude=5.0, float_speed=0.09,
                wisp_cycle=120, wisp_offset=80, wisp_mode="high",
            ),

            # =========================================================================
            # LAYER 2: MID-SKY TIER (y = 75 -> 145) - Mid-Altitude Wisps & Sweeping Lasers
            # =========================================================================
            WispGhost(
                x=540.0, y=125.0, patrol_min_x=480.0, patrol_max_x=600.0,
                speed=-1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=50, wisp_mode="alternating",
            ),
            # Mid-Altitude LaserGhost 1 (over Zone 1 Graves)
            LaserGhost(
                x=800.0, y=130.0, patrol_min_x=730.0, patrol_max_x=870.0,
                speed=1.2, float_amplitude=6.0, float_speed=0.07,
                laser_length=85.0, laser_rotation_speed=0.037,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.85,
            ),
            WispGhost(
                x=1400.0, y=110.0, patrol_min_x=1340.0, patrol_max_x=1450.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=25, wisp_mode="alternating",
            ),
            # Mid-Altitude LaserGhost 2 (over Zone 2 Abyss)
            LaserGhost(
                x=2040.0, y=75.0, patrol_min_x=1990.0, patrol_max_x=2100.0,
                speed=-1.1, float_amplitude=5.0, float_speed=0.06,
                laser_length=65.0, laser_rotation_speed=0.036,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.75,
            ),
            WispGhost(
                x=2450.0, y=85.0, patrol_min_x=2370.0, patrol_max_x=2530.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=15, wisp_mode="alternating",
            ),
            WispGhost(
                x=3150.0, y=145.0, patrol_min_x=3070.0, patrol_max_x=3230.0,
                speed=1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=75, wisp_mode="alternating",
            ),
            # Mid-Altitude LaserGhost 3
            LaserGhost(
                x=3620.0, y=120.0, patrol_min_x=3540.0, patrol_max_x=3680.0,
                speed=-1.2, float_amplitude=6.0, float_speed=0.07,
                laser_length=80.0, laser_rotation_speed=0.038,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.85,
            ),
            WispGhost(
                x=4050.0, y=135.0, patrol_min_x=3970.0, patrol_max_x=4130.0,
                speed=1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=30, wisp_mode="alternating",
            ),
            WispGhost(
                x=4400.0, y=140.0, patrol_min_x=4320.0, patrol_max_x=4480.0,
                speed=-1.4, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=90, wisp_mode="alternating",
            ),
            WispGhost(
                x=4900.0, y=145.0, patrol_min_x=4820.0, patrol_max_x=4980.0,
                speed=1.3, float_amplitude=6.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=40, wisp_mode="alternating",
            ),
            # Mid-Altitude LaserGhost 4
            LaserGhost(
                x=5300.0, y=75.0, patrol_min_x=5220.0, patrol_max_x=5380.0,
                speed=-1.2, float_amplitude=5.0, float_speed=0.06,
                laser_length=65.0, laser_rotation_speed=0.035,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.75,
            ),
            WispGhost(
                x=5750.0, y=85.0, patrol_min_x=5670.0, patrol_max_x=5830.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=70, wisp_mode="alternating",
            ),
            WispGhost(
                x=6350.0, y=100.0, patrol_min_x=6270.0, patrol_max_x=6430.0,
                speed=-1.4, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=20, wisp_mode="alternating",
            ),
            WispGhost(
                x=6700.0, y=120.0, patrol_min_x=6620.0, patrol_max_x=6710.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=55, wisp_mode="alternating",
            ),

            # Mid-Altitude Lethal Rotating & Sweeping Laser Ghosts
            LaserGhost(
                x=510.0, y=165.0, patrol_min_x=480.0, patrol_max_x=550.0,
                speed=1.0, float_amplitude=5.0, float_speed=0.07,
                laser_length=75.0, laser_rotation_speed=0.035,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.80,
            ),
            LaserGhost(
                x=2800.0, y=160.0, patrol_min_x=2740.0, patrol_max_x=2860.0,
                speed=1.0, float_amplitude=6.0, float_speed=0.07,
                laser_length=90.0, laser_rotation_speed=0.036,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.80,
            ),
            LaserGhost(
                x=3850.0, y=170.0, patrol_min_x=3790.0, patrol_max_x=3910.0,
                speed=-1.0, float_amplitude=6.0, float_speed=0.07,
                laser_length=95.0, laser_rotation_speed=0.037,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.85,
            ),
            LaserGhost(
                x=4750.0, y=190.0, patrol_min_x=4700.0, patrol_max_x=4800.0,
                speed=1.1, float_amplitude=6.0, float_speed=0.06,
                laser_length=85.0, laser_rotation_speed=0.036, laser_initial_angle=0.0,
            ),
            LaserGhost(
                x=4870.0, y=190.0, patrol_min_x=4820.0, patrol_max_x=4920.0,
                speed=-1.1, float_amplitude=6.0, float_speed=0.06,
                laser_length=85.0, laser_rotation_speed=-0.036, laser_initial_angle=3.14,
            ),
            LaserGhost(
                x=6640.0, y=130.0, patrol_min_x=6610.0, patrol_max_x=6670.0,
                speed=-1.1, float_amplitude=5.0, float_speed=0.07,
                laser_length=75.0, laser_rotation_speed=0.036,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.80,
            ),

            # =========================================================================
            # LAYER 3: LOW/MID-AIR TIER (y = 175 -> 190) - Safely Above Ground Tombstones
            # =========================================================================
            WispGhost(
                x=600.0, y=185.0, patrol_min_x=520.0, patrol_max_x=680.0,
                speed=1.2, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=30, wisp_mode="low",
            ),
            # Low Ground LaserGhost 1
            LaserGhost(
                x=1020.0, y=185.0, patrol_min_x=950.0, patrol_max_x=1070.0,
                speed=-1.2, float_amplitude=5.0, float_speed=0.07,
                laser_length=75.0, laser_rotation_speed=0.035,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.75,
            ),
            WispGhost(
                x=2350.0, y=130.0, patrol_min_x=2290.0, patrol_max_x=2410.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=20, wisp_mode="low",
            ),
            WispGhost(
                x=3000.0, y=185.0, patrol_min_x=2920.0, patrol_max_x=3080.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=80, wisp_mode="low",
            ),
            WispGhost(
                x=3500.0, y=185.0, patrol_min_x=3420.0, patrol_max_x=3580.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=130, wisp_offset=40, wisp_mode="low",
            ),
            # Low Ground LaserGhost 2
            LaserGhost(
                x=3900.0, y=185.0, patrol_min_x=3830.0, patrol_max_x=3970.0,
                speed=-1.1, float_amplitude=5.0, float_speed=0.07,
                laser_length=75.0, laser_rotation_speed=0.036,
                laser_initial_angle=1.57, laser_sweep_amplitude=0.75,
            ),
            WispGhost(
                x=4700.0, y=185.0, patrol_min_x=4620.0, patrol_max_x=4780.0,
                speed=1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=135, wisp_offset=70, wisp_mode="low",
            ),
            WispGhost(
                x=6200.0, y=185.0, patrol_min_x=6120.0, patrol_max_x=6280.0,
                speed=-1.3, float_amplitude=5.0, float_speed=0.08,
                wisp_cycle=140, wisp_offset=50, wisp_mode="low",
            ),

            # =========================================================================
            # SLIDE TUNNEL MOUTH GUARDS (Standard Ghost) - Floating at y=272.0
            # Challenges player to CROUCH-SLIDE under the ghost directly into the tunnel!
            # =========================================================================
            # Guard in front of Tunnel 1 (x=1080)
            Ghost(
                x=1010.0, y=DEFAULT_FLOOR_Y - 48.0, patrol_min_x=970.0, patrol_max_x=1055.0,
                speed=1.4, float_amplitude=4.0, float_speed=0.08,
            ),
            # Guard in front of Tunnel 2 (x=2740)
            Ghost(
                x=2670.0, y=DEFAULT_FLOOR_Y - 48.0, patrol_min_x=2620.0, patrol_max_x=2715.0,
                speed=-1.5, float_amplitude=4.0, float_speed=0.08,
            ),
            # Guard in front of Tunnel 3 (x=4480)
            Ghost(
                x=4410.0, y=DEFAULT_FLOOR_Y - 48.0, patrol_min_x=4360.0, patrol_max_x=4455.0,
                speed=1.5, float_amplitude=4.0, float_speed=0.08,
            ),
            # Guard in front of Tunnel 4 (x=6080)
            Ghost(
                x=6010.0, y=DEFAULT_FLOOR_Y - 48.0, patrol_min_x=5960.0, patrol_max_x=6055.0,
                speed=-1.6, float_amplitude=4.0, float_speed=0.08,
            ),
            # Outdoor Sanctuary Guard (Guards airborne space above Pit 8, leaving runway x=6880..6940 clear)
            Ghost(
                x=6840.0, y=DEFAULT_FLOOR_Y - 75.0, patrol_min_x=6800.0, patrol_max_x=6875.0,
                speed=1.5, float_amplitude=5.0, float_speed=0.08,
            ),
        ],
        items=[
            WarmthItem(x=312.0, y=DEFAULT_FLOOR_Y - 138.0),   # Atop Stair 1 Summit Pillar
            WarmthItem(x=1140.0, y=DEFAULT_FLOOR_Y - 18.0),  # INSIDE TUNNEL 1 (Slide reward!)
            WarmthItem(x=1520.0, y=DEFAULT_FLOOR_Y - 140.0), # Atop Sentry Platform 1 Altar
            WarmthItem(x=1800.0, y=DEFAULT_FLOOR_Y - 240.0), # Atop High Sky Catwalk
            WarmthItem(x=2464.0, y=DEFAULT_FLOOR_Y - 165.0), # Atop Temple Pyramid Apex
            WarmthItem(x=2810.0, y=DEFAULT_FLOOR_Y - 18.0),  # INSIDE TUNNEL 2 (Slide reward!)
            WarmthItem(x=3740.0, y=DEFAULT_FLOOR_Y - 165.0), # Atop Valley Pagoda Pillar
            WarmthItem(x=4125.0, y=DEFAULT_FLOOR_Y - 125.0), # Atop Stepping Altar
            WarmthItem(x=4550.0, y=DEFAULT_FLOOR_Y - 18.0),  # INSIDE TUNNEL 3 (Slide reward!)
            WarmthItem(x=5480.0, y=DEFAULT_FLOOR_Y - 215.0), # Atop Sky Citadel Pillar
            WarmthItem(x=6145.0, y=DEFAULT_FLOOR_Y - 18.0),  # INSIDE TUNNEL 4 (Slide reward!)
            WarmthItem(x=6498.0, y=DEFAULT_FLOOR_Y - 175.0), # Final Apex Summit Altar
            WarmthItem(x=7030.0, y=DEFAULT_FLOOR_Y - 110.0), # Floating above Sacred Sanctuary Tablet Monument (y=210)
        ],
        goal=GoalFlag(x=7150.0, y=DEFAULT_FLOOR_Y - 88.0),
        floor_y=DEFAULT_FLOOR_Y,
        max_steps=20000,
    )


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
    4: _build_level_4,
    5: _build_level_5,
    6: _build_level_6,
    7: _build_level_7,
    8: _build_level_8,
}


def get_level(level_id: int, seed: Optional[int] = None) -> LevelConfig:
    """
    Return a LevelConfig for the requested level_id.
    If seed is provided (for Levels 1-7), synthesizes a procedural fair instance.
    If seed is None, returns the handcrafted default curriculum stage.
    """
    if seed is not None and 1 <= level_id <= 7:
        from .level_generator import LevelGenerator
        return LevelGenerator.generate(level_id=level_id, seed=seed)

    if level_id not in LEVEL_BUILDERS:
        available = list(LEVEL_BUILDERS.keys())
        raise ValueError(f"Unknown level_id={level_id}. Available levels: {available}")
    return LEVEL_BUILDERS[level_id]()


def create_env(level_id: int = 1, seed: Optional[int] = None) -> GameEnvironment:
    """
    Build a GameEnvironment configured with the chosen level.
    Accepts an optional seed for fair procedural blind testing on Levels 1-7.
    """
    cfg_level = get_level(level_id, seed=seed)
    return GameEnvironment(
        obstacles=cfg_level.obstacles,
        pits=cfg_level.pits,
        goal=cfg_level.goal,
        ghosts=cfg_level.ghosts,
        monsters=cfg_level.monsters if cfg_level.monsters is not None else [],
        items=cfg_level.items,
        floor_y=cfg_level.floor_y,
        max_steps=cfg_level.max_steps,
        level_name=cfg_level.name,
        level_id=level_id,
    )
