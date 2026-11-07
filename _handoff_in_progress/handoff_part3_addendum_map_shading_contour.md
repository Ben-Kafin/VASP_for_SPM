# Part 3 addendum: shading and contour for the Map view's LDOS maps

Repository: `VASP_for_SPM`, the simulator package `LDOS_spread_decomp/stm_from_dft_opt/stm_from_dft/`.

Part 3's rules 1, 3, 5 and 7 apply (rule 2 was dropped by the owner's ruling 9). The `simulator.py` edits wait with part 3's, until part 2's workflow has finished. This goes with part 3 for the same reasons as the two colour-map sets (response document, section 6): part 3 is not yet implemented and already changes the simulator's window, and the new box's controls show in part 3's info panel settings section with no extra work.

Nothing here waits on the owner. Points marked "Claude's reading" are readings of the owner's words, recorded as such.

## 1. One portable module (the owner's rule)

The owner: "this whole cet cmap and shading and contour toggle options I will want to import into the sm4 eventually so I would like that to be its own ortable zip or wtv. but for now I just want a document I can give the code session about this."

- **A new module** in the simulator package holds all of it:
  - the two colour-map sets of the response document's section 6;
  - the shading of this document;
  - the Contour / Shading box and its rules.

  Its name is the code session's choice unless the owner names one.
- **Self-contained:** it imports nothing from `simulator.py`. The simulator passes in what it needs (the maps' values, the LDOS topo heights, the cell grid's geometry, the colour norm) and gets back colours, colour-map names and the box's state. It then can be copied into sm4_Browser later. Each package keeps its own copy, as the owner's rule for shared logic says.
- **No zip now.** The module only has to be written so it can be packaged on its own later.
- **Its maths and the GPU, Claude's reading:** the owner's rule puts the simulator's computation on the GPU.
  - The module's array maths works on whichever array type it is given, NumPy or CuPy (`cupy.get_array_module`).
  - The simulator passes CuPy arrays, so the shading runs on the GPU there, and the same code runs on NumPy in sm4_Browser.

## 2. The box (the owner's rule)

The owner: "add the contour toggle option and the shading toggles as their own box. I also want to be able to turn them both off."

- **A box of its own** with:
  - a **Contour** toggle;
  - a **Shading** toggle;
  - the **Height / Relief** choice for the shading.

  **Where (Claude's reading):** it is shown in Map only, as the `cmap norm` box is today (`_strip_boxes['cmap norm'].setVisible(mode == 'Map')`).
- **At most one on, both can be off:**
  - turning Shading on turns Contour off (the owner's design);
  - turning Contour on turns Shading off (Claude's reading: both on is not in the owner's design, and "both off" is the added case);
  - either can be turned off, leaving both off.
- **Contour today:** there is no contour toggle; in Map the contour slice is always drawn (`_draw_contour_artists`, the level line on the topo colour bar).
  - With the toggle on, the contour slice is as today.
  - With it off, no traces are drawn on the energy maps, and the level line on the topo colour bar is hidden (Claude's reading).
- **At start (Claude's reading):** Contour on and Shading off, so the window opens as it does today. The Height / Relief choice starts at Height, the owner's first description.
- **Changes are display only:** no LDOS is recomputed and nothing is rearranged (part 1 §5.2). Only the energy maps are redrawn.

## 3. Shading (the owner approved this setup)

**What is shaded:** the Map view's LDOS energy maps only. The owner: "only the ldos maps."

**The colour map while Shading is on:** the energy maps switch from the set's Map map (`MAP_CMAP` in default 1, `'CET-L6'` in default 2) to an isoluminant map, so that lightness carries only the shading and hue carries the value.
- **The LDOS:** `'CET-I1'` (the owner: "use i1"). Blue to green to orange at CIELAB L* about 70.
- **Mag, signed spin density** (collinear spin-polarized runs): `'CET-D11'` (the owner). It is diverging and isoluminant, blue to gray to orange at L* about 70.
- **Mag, |m|** (NCL / SOC runs, never negative): `'CET-I3'` (the owner: "use i3"). It is isoluminant, blue to pink at L* about 70, the same lightness as CET-I1 and CET-D11. It replaces the owner's first choice, `'CET-L19'`, which is not isoluminant.
- **Turning Shading off:** the set's maps come back (Claude's reading).

**Height:** the brightness comes from the LDOS topo height: darker where low, lighter where high. The owner: "I sorta just wanted the low areas darker and the high area lighter".

**Relief:** the brightness comes from the LDOS topo's slopes, lit from one direction.
- **Kovesi's defaults (the owner: "use kovesi's defaults"):** the shading of Kovesi's `relief.m` (CET's MATLAB code, 2014) with its defaults:
  - azimuth 45° (0 upwards, increasing clockwise, so the light comes from the upper right) and elevation 45°;
  - gradient scale 1;
  - the light vector `[cos(el) sin(az), cos(el) cos(az), sin(el)]`, made a unit vector;
  - Lambertian shading: the dot product of the unit surface normal and the light vector;
  - values below `minval`, default 0, set to 0 (surfaces facing away from the light).

  This replaces matplotlib's 315° in the earlier setup.
- **The normals:** the cell's grid is the cell's fractional grid times its vectors (`grid_xy`), which need not be at right angles.
  - The normals come from the gradients along the two cell vectors, converted to x and y.
  - Each normal is made a unit vector before the dot product.

**The combination (Kovesi's method, as the owner's source describes it):** the colour is multiplied by the brightness, never blended with it as a weighted sum.

**The strength, set from the data (the owner):** "whatever intensity uses the perception ability calculated in the CET color maps to highlight the min and max and all the differences adequately." As approved:
- The scale is CIELAB L*, the perceptual lightness the CET maps are built on.
- Each point's colour keeps its hue, and its L* is scaled by a factor that runs evenly from a floor fraction, at the lowest shade value over the shown area, to 1 at the highest. The lowest shade value is the lowest height in Height and the darkest slope in Relief.
  - For CET-I1, CET-D11 and CET-I3, every colour starts at L* about 70, so the shades run evenly from 70 times the floor up to 70.
- **The floor: 0, Kovesi's `minval` default** (the owner: "use kovesi's defaults"). Claude found no published floor for a stretched shade; `relief.m`'s `minval` is the nearest published default.
  - **Claude's reading, recorded as such:** the approved stretch stays, with Kovesi's defaults inside it. So the lowest shade value over the shown area is black in both modes, and no hue shows there.
  - The other reading is Kovesi's `relief.m` exactly, with no stretch. There, only surfaces facing away from the light go black. The owner was told of both.
- **Exact even steps:** each point's colour is scaled to its target L*. That means the target L* is converted to relative luminance, linear RGB is scaled by the target over the colour's own luminance, and the result goes back to sRGB.
- **In Relief,** this stretch takes the place of a hand-set intensity or height exaggeration.
- **The height range:** the LDOS topo's lowest to highest point over the shown area, as the contour slice's level already uses the topo's height range (`z_frac`).

**The other display options keep working:** the colour map is applied first, with the maps' own norm (LDOS γ, `cmap norm`, Norm, the decomp partitions), and the shading is applied after it.

**The colour bars (Claude's reading):** they show the isoluminant map unshaded. Hue reads the value from them; the shading is not in the bar.

**Drawing:**
- Shading needs one colour per grid point, so while Shading is on the energy maps are drawn as a coloured mesh on the cell's grid instead of filled contours.
- `qtgui`'s quad item already takes one RGBA colour per quad in data units (what `pcolormesh` uses), so the slanted cell grid is drawn as it is.
- **The resolution (the owner):** "the ldos topo and map heatmaps will already have the same resolution in both the simulator and sm4 browser". The shading is done at the grid's own points, with no interpolation; each grid cell is one colour.

**The saves:** the shading is in the colours, so every save shows it: PNG, TIFF, the Qt-drawn SVG and the matplotlib vector saves.

## 4. Not changed

The contour slice's own drawing (`contour.py`, `CONTOUR_COLOR`, `CONTOUR_LEVEL_COLOR`), the topo panels, the Line / Path heatmaps, and every other window.
