# Response document: N1 to N4

The owner's answers to `PART3_OPEN2.md` (N1 to N4). Nothing here waits on the owner.

**N1, the Line view's height sharing (P2): (a), the owner's answer.** In `qtgui`, in both package copies:
- `controls.Arrangement` takes a list of allowed layouts, and Line passes only `row_top`, the owner's two rows.
- A row of fill panels reports, as its shape, the height at which its panels fill the row's width.

Then the top row (global topo, Line Topo, stripe, heatmaps) grows with the window, keeping its ratios, up to what the
width allows. The 1D plots take the rest, as P2 says.

**N2, the Qt-drawn "as shown" SVG: (c), the owner's answer.** 720 dpi with the text in proportion, in all three copies.
- `export_vector` writes the SVG at 720 dpi (`gen.setResolution(720)`), at the same physical size as the 720 dpi window
  images.
- While the file is drawn, every font is given as a pixel size equal to its size on screen. Qt's SVG writer (qtsvg 6.8,
  `qfontToSvg`) writes a pixel-size font as it is. A point-size font it writes as `pointSizeF * resolution / 72`, which
  at 720 would make every text about 7.5 times larger than the panels.
- So the text and the panels keep the same sizes relative to each other as on screen, the owner's rule for every 720
  dpi file.
- When the file is written, every font goes back to its point size, so the window is unchanged.

**N3, the vector saves after a colour-set change: (a), the owner's answer.** `TriContourSet.set_cmap` and `set_norm`
also update the record `mplexport` rebuilds the filled contours from (`TriContourSet._args`), in `qtplot`, both package
copies. A PDF, EPS or SVG save then shows the colour maps on screen.

**N4, README rows for the new modules: as recommended, the owner agreed.**
- A row for `cmap_shading.py` in the stm README's module table.
- A row for `mo_ldos_projection/_resolve.py` in the mo_ldos_decomp README, so the README no longer reads as if
  `_imports.py` defined the resolvers.

Both are in the same terms as K9's row for `ldos_decomp.py`.
