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

---

## 3. Component Visuals & Hitboxes

- **Window & Camera**: `800 x 400` px logic canvas scaled smoothly to `1400 x 700` window display. `DEFAULT_FLOOR_Y = 320.0`. Camera tracks Hoá keeping him ~30% from left edge (`camera_x = max(0, x - width * 0.3)`).
- **Hoá (`Player` — `24 x 32` px)**: Áo Ngũ Thân robe, Khăn Đóng turban, bamboo lantern with dynamic red tassel and ambient aura, white trousers, black shoes. Hitbox shrinks to `24 x 18` px when crouching/sliding.
- **Tombstone (`Obstacle` — `32 x [44..92]` px)**: Procedural pixel art rendering supporting `"cross"`, `"buddhist"`, and `"platform"` styles.
- **Open Grave Pit & Flame (`Pit` — `65..85` px wide)**:
  - Deep grave trench with grounded magma bed.
  - Active lethal flame jets with variable heights (`fire_height`, default `65.0px`).
- **Spectral Ghost (`Ghost` — `24 x 28` px)**: Spectral translucent trail, rounded dome, wavy tails, dark hollow eyes with specular glint. Opt-in rotating eye-laser (`has_laser=True`).
- **Warmth Item (`WarmthItem` — `14 x 14` px)**: Golden ember orb with halo ring.
- **Khuê Văn Các Pavilion (`GoalFlag` — `64 x 88` px)**: Four stone pillars, three-tier courtyard steps, illuminated sunburst window, lacquered gold board ("奎文閣"), double-tiered curved roofs.

---

## 4. Curriculum Progression & Stages (Levels 1–5)

1. **Level 1 (`Quiet Graveyard`)**: 2 tombstones, 0 pits, 0 ghosts, 0 items (`max_steps=800`). Baseline jumping.
2. **Level 2 (`Open Graves Pits`)**: 3 tombstones + 2 grave pits, 0 ghosts, 0 items (`max_steps=1000`). Timed gap leaps.
3. **Level 3 (`Whispering Graves`)**: 3 tombstones + 2 pits (including 1 hellfire pit) + 2 uniform ghosts (jump-over & walk-under) + 3 warmth items (`max_steps=1000`).
4. **Level 4 (`Nightmare Catacombs`)**: 5 tombstones + 4 hellfire pits + 5 diverse ghosts (including dual counter-rotating laser scissors) + 5 warmth items (`max_steps=1400`). Continuous 360-degree rotating laser hazards.
5. **Level 5 (`The Eternal Abyss`)**: Marathon gauntlet with variable flame heights, multi-speed laser ghosts, varied tombs, and mandatory crouch tunnels (`max_steps=2400`).
