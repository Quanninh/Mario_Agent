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


LEVEL_BUILDERS: Dict[int, Callable[[], LevelConfig]] = {
    1: _build_level_1,
    2: _build_level_2,
    3: _build_level_3,
    4: _build_level_4,
    5: _build_level_5,
    6: _build_level_6,
    7: _build_level_7,
}


def get_level(level_id: int) -> LevelConfig:
    """Return a fresh LevelConfig for the requested level_id."""
    if level_id not in LEVEL_BUILDERS:
        available = list(LEVEL_BUILDERS.keys())
        raise ValueError(f"Unknown level_id={level_id}. Available levels: {available}")
    return LEVEL_BUILDERS[level_id]()


def create_env(level_id: int = 1) -> GameEnvironment:
    """Build a GameEnvironment configured with the chosen level."""
    cfg_level = get_level(level_id)
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
    )
