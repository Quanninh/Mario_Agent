# Visual & Game Design Spec — Hoá's Midnight Escape

> Comprehensive theme, color palettes, pixel-art visual components, and level curriculum specifications for Member 1 (`env/`).

---

## 1. Story & Creative Theme
- **Setting**: A cold, haunted midnight cemetery infused with traditional Vietnamese cultural heritage and cosmic folklore aesthetics.
- **Protagonist**: **Hoá**, dressed in traditional Vietnamese scholar attire (**Áo Ngũ Thân May Kép Đệm Bông** and **Khăn Đóng**), carrying a swinging handheld **Bamboo Lantern**. Stranded in the cemetery late at night, he must navigate hazards to reach sanctuary.
- **Sanctuary Finish**: Hanoi's iconic **Khuê Văn Các** (Temple of Literature Pavilion) with carved stone pillars, flared terracotta roofs, and glowing circular sunburst window (*Cửa Sổ Sao Khuê*).
- **Visual Style**: Authentic 16-bit retro pixel art with ordered Bayer dithering, multi-layered parallax scroll clouds, and custom procedural shading.

---

## 2. Color Palettes (RGB)

### 2.1 16-Bit Dithered Twilight Night Sky (Bayer 4x4 Matrix)
| Band | RGB Value | Visual Role |
|:---|:---|:---|
| **Band 0** | `(16, 10, 24)` | Deep Obsidian Void (Top) |
| **Band 1** | `(26, 14, 38)` | Deep Midnight Violet |
| **Band 2** | `(38, 20, 56)` | Dark Royal Aubergine |
| **Band 3** | `(52, 28, 78)` | Twilight Amethyst |
| **Band 4** | `(70, 44, 92)` | Muted Dusk Purple |
| **Band 5** | `(92, 64, 110)` | Soft Mauve Twilight |
| **Band 6** | `(102, 72, 120)` | Warm Horizon Twilight Glow (Bottom) |

### 2.2 Crimson Blood Moon & All-Seeing Eye (ॐ Om Rune)
| Element | RGB Value | Role |
|:---|:---|:---|
| **Cosmic Aura** | `(35, 18, 28)` $\to$ `(92, 24, 36)` | Multi-tiered ethereal moon glow rings |
| **Blood Moon Core** | `(150, 26, 32)` $\to$ `(235, 75, 75)` | Deep crimson lunar sphere with shading & craters |
| **Ancient Tree Branches** | `(24, 20, 22)` $\to$ `(90, 75, 65)` | Gnarled dark tree branches entwining the moon |
| **All-Seeing Eye Sclera** | `(244, 238, 220)` | Almond-shaped sacred central eye |
| **Sacred Rune (ॐ Om)** | `(34, 26, 24)` | Sacred Om glyph in eye pupil |

### 2.3 Traditional Vietnamese Scroll Clouds (*Vân Mây Ruyi*)
| Element | RGB Value | Role |
|:---|:---|:---|
| **Deep Plum Border** | `(32, 18, 42)` | Sharp outer contour for strong separation against sky |
| **Orchid Shadow** | `(138, 108, 158)` | Underside cloud lobe shading |
| **Luminous Lilac Body** | `(212, 192, 232)` | High-luminance cloud body that pops in the night |
| **Moonlit Ivory Crest** | `(252, 248, 255)` | Specular highlights on top lobe edges |
| **Filigree Spirals** | `(56, 32, 72)` | Concentric inner whorls & trailing ribbon curves |

### 2.4 Tombstones & Floating Platforms
| Style | Key Colors | Architecture & Visual Characteristics |
|:---|:---|:---|
| **Catholic Tomb** (`style="cross"`) | `(46, 54, 66)` body, `(195, 210, 228)` bevel, `(232, 198, 75)` gold, `(62, 108, 66)` turf | Dominant Latin Cross (>60% total height) with 3D bevels, standing on an upright gold-framed stele and stepped granite slab with central green grass turf. |
| **Traditional Tomb** (`style="buddhist"`) | `(125, 138, 152)` stone, `(255, 245, 160)` gold, `(220, 60, 45)` incense | Carved lotus bud finial, 2-tier curved roofs (*mái đao cong vảy rồng*), dragon pillars, radiant golden Sauwastika (*CHỮ VẠN 卍*), and incense burner. |
| **Floating Platform** (`style="platform"`) | `(125, 138, 152)` stone, `(185, 198, 212)` highlight, `(56, 84, 60)` moss | Weathered stone masonry blocks with 3D bevel highlights, mortise brick seams, and moss accents. |

### 2.5 Laser Ghost (*Ma Đội Nón Lá Bắn Laser*)
| Element | RGB Value | Role |
|:---|:---|:---|
| **Conical Hat (*Nón Lá*)** | `(166, 128, 70)` body, `(210, 180, 110)` rim, `(71, 51, 24)` shadow | Traditional Vietnamese straw hat with weave texture |
| **Daoist Talisman (*Bùa Lệnh*)** | `(245, 210, 65)` parchment, `(199, 34, 34)` cinnabar rune | Yellow warding paper inscribed with crimson spell glyph |
| **Bagua Mirror (*Kính Bát Quái*)** | `(255, 179, 38)` gold frame, `(255, 45, 75)` ambient halo | Octagonal brass mirror on chest warding wandering spirits |
| **Spectral Body** | `(140, 26, 52)` midtone, `(204, 41, 77)` highlight, `(74, 14, 28)` shadow | Blood-crimson phantom silhouette contrasting with night sky |
| **Rotating Laser Beam** | `(255, 30, 90)` core, `(255, 80, 160)` outer glow | 360° continuously rotating lethal beam originating from eyes |

### 2.6 Wisp Maiden (*Bạch Y Ma Nữ*) & Will-o'-the-Wisp
| Element | RGB Value | Role |
|:---|:---|:---|
| **Spectral Hair** | `(28, 184, 150)` mid, `(61, 245, 202)` cyan glow, `(171, 255, 240)` rim | Long flowing emerald/cyan ethereal tresses |
| **Mourning Robe** | `(240, 246, 255)` body, `(195, 212, 235)` mid, `(110, 130, 155)` drape | Flowing white spectral robe drifting in the wind |
| **Charging Spirit Flame** | `(60, 245, 210)` core, `(180, 255, 240)` white-hot highlight | Handheld charging flame 25 ticks prior to casting |
| **Will-o'-the-Wisp Projectile** | `(40, 235, 200)` body, `(220, 255, 248)` core, `(255, 215, 60)` sparks | Airborne teardrop fireball with trailing and ground-rolling sparks |

### 2.7 Warding Garlic Bulb (*Củ Tỏi Trừ Tà*) & Collection Burst
| Element | RGB Value | Role |
|:---|:---|:---|
| **Clove Body** | `(216, 195, 215)` lilac mid, `(138, 92, 130)` violet shadow, `(242, 235, 245)` ivory | Volumetric bulbous cloves with soft crevice contouring |
| **Specular Gleam** | `(255, 255, 255)` pure white | Radiant vertical gleam down the central clove |
| **Sickle Hook Stalk** | `(188, 148, 96)` tan body, `(235, 205, 145)` highlight crest | Graceful rightward curved straw stalk |
| **Woody Root Base** | `(135, 95, 50)` dark root, `(185, 145, 90)` highlight | Compact, subtle root nub at bottom |
| **Dotted Golden Aura & Embers**| `(250, 225, 140)` aura dots, `(255, 230, 130)` / `(255, 190, 70)` embers | Form-fitting golden halo with floating orbiting ember pixels |
| **Collection Burst** | `(235, 228, 245)` $\to$ `(150, 140, 172)` smoke, `(255, 215, 80)` stars | Billowing lilac smoke clouds and radiating 4-pointed sparkle stars |

---

## 3. Component Visuals & Hitboxes

- **Window & Camera**: `800 x 400` px logic canvas scaled smoothly to `1400 x 700` window display. `DEFAULT_FLOOR_Y = 320.0`. Camera tracks Hoá keeping him ~30% from left edge (`camera_x = max(0, x - width * 0.3)`).
- **Hoá (`Player` — `24 x 32` px)**: Áo Ngũ Thân robe, Khăn Đóng turban, bamboo lantern with dynamic red tassel and ambient aura, white trousers, black shoes. Hitbox shrinks to `24 x 18` px when crouching/sliding.
- **Tombstone (`Obstacle` — `32 x [44..92]` px)**: Procedural pixel art rendering supporting `"cross"`, `"buddhist"`, and `"platform"` styles.
- **Open Grave Pit & Flame (`Pit` — `65..85` px wide)**:
  - Deep grave trench with grounded magma bed.
  - Active lethal flame jets with variable heights (`fire_height`, default `65.0px`).
- **Base Ghost (`Ghost` — `24 x 28` px)**: Spectral translucent trail, rounded dome, wavy tails, dark hollow eyes with specular glint.
- **Laser Ghost (`LaserGhost` — `24 x 28` px)**: Nón Lá straw hat, talisman, chest Bagua mirror, blood-crimson aura, and lethal 360° rotating laser beam (length `90.0px`).
- **Wisp Maiden (`WispGhost` — `22 x 26` px)**: Flowing cyan hair, white spectral mourning robe, charging spirit flame, and periodic will-o'-the-wisp casting.
- **Will-o'-the-Wisp (`WillOWisp` — `8 x 8` px)**: Cyan spirit fireball projectile with parabolic arcs, ground bounce, friction, and rolling embers.
- **Warding Garlic Bulb (`WarmthItem` — `16 x 16` px)**: Authentic Vietnamese sacred garlic bulb with curved sickle hook stalk, soft lilac shading, pure white central gleam, dotted golden aura, and collected burst animation.
- **Khuê Văn Các Pavilion (`GoalFlag` — `64 x 88` px)**: Four stone pillars, three-tier courtyard steps, illuminated sunburst window, lacquered gold board ("奎文閣"), double-tiered curved roofs.

---

## 4. Curriculum Progression & Stages (Levels 1–5)

1. **Level 1 (`Quiet Graveyard`)**: 2 tombstones, 0 pits, 0 ghosts, 0 items (`max_steps=800`). Baseline jumping.
2. **Level 2 (`Open Graves Pits`)**: 3 tombstones + 2 grave pits, 0 ghosts, 0 items (`max_steps=1000`). Timed gap leaps.
3. **Level 3 (`Whispering Graves`)**: 3 tombstones + 2 pits (including 1 hellfire pit) + 2 uniform ghosts (jump-over & walk-under) + 3 garlic bulbs (`max_steps=1000`).
4. **Level 4 (`Nightmare Catacombs`)**: 5 tombstones + 4 hellfire pits + 5 diverse ghosts (including dual counter-rotating laser scissors) + 5 garlic bulbs (`max_steps=1400`). Continuous 360-degree rotating laser hazards.
5. **Level 5 (`Valley of the Wisp Maidens`)**: Intermediate-hard vertical gauntlet featuring 18 ghosts: 8 high-sky and mid-air Wisp Maidens hovering between $y = 70$ and $y = 155$, 5 ground-level LaserGhosts sweeping 360° rotating beams, 2 ground Wisp Maidens, 3 patrolling base ghosts, 7 tombstones, 5 pits (including 3 hellfire eruptions), and 6 garlic bulbs (`max_steps=2400`).
