"""
mplexport: a qtplot figure (what a window shows) rebuilt as a matplotlib figure and saved by
matplotlib, for publication files: PDF, EPS, SVG (vector, every line and text an editable object,
e.g. in Adobe Illustrator), and PNG / TIFF at a chosen dpi.

matplotlib is used here and only here, at the moment of saving: the windows draw and interact
with pyqtgraph (fast marker dragging, panning, many curves), and matplotlib's renderers make
the file.  The rebuild reads what qtplot already records in matplotlib's terms: every axes at its
figure-fraction position with its current limits, labels, title, ticks, aspect, and legend; every
curve with its style; images with their colour map and norm (and an affine placement); quad
meshes as filled polygons; filled contours (contourf and tricontourf from the arguments they were
made with); scatter collections with their colours and sizes; reference lines, spans, line
collections (a colour per segment kept), texts, annotations, colour bars, legend proxies (Line2D,
Patch), and the figure's texts.  The controls (buttons, boxes) are not part of the figure, nor
are the readout texts in the control strip below the panels, hidden axes, or hidden artists.

The matplotlib figure has the window's design size (the window's DESIGN constant, at qtplot.DPI),
whatever the window's size on screen, and every text its point size as set (the window's text
zoom, qtplot.window_zoom, is not applied); the panels keep the figure fractions they have on
screen.  The file is cropped to what is drawn (savefig with bbox_inches='tight').

    export(fig, path, dpi=300)       the figure into `path` (format from the extension)
    to_matplotlib(fig)               the matplotlib Figure itself (to adjust before saving)
    available()                      whether matplotlib can be imported (qtplot.Figure.savefig
                                     falls back to Qt's own writers when it cannot)
"""

import numpy as np

from qtgui import qtplot as q


def available():
    """True when matplotlib can be imported (the publication formats need it)."""
    try:
        import matplotlib            # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------- #
#  Conversions                                                             #
# ---------------------------------------------------------------------- #

def _mpl_cmap(cm):
    """A qtplot colour map (anything to_cmap accepts) as a matplotlib ListedColormap of the same table."""
    from matplotlib.colors import ListedColormap
    cm = q.to_cmap(cm)
    return ListedColormap(cm._table, name=cm.name)


def _mpl_norm(norm):
    """
    A qtplot norm as the matching matplotlib norm (PowerNorm, TwoSlopeNorm, LogNorm, SymLogNorm,
    else Normalize); a norm of a window's own that gives to_matplotlib() is asked for its
    matplotlib form; None stays None.
    """
    import matplotlib.colors as mc
    if norm is None:
        return None
    if hasattr(norm, 'to_matplotlib'):
        return norm.to_matplotlib()
    if isinstance(norm, q.PowerNorm):
        return mc.PowerNorm(norm.gamma, vmin=norm.vmin, vmax=norm.vmax)
    if isinstance(norm, q.TwoSlopeNorm):
        return mc.TwoSlopeNorm(norm.vcenter, vmin=norm.vmin, vmax=norm.vmax)
    if isinstance(norm, q.LogNorm):
        return mc.LogNorm(vmin=norm.vmin, vmax=norm.vmax)
    if isinstance(norm, q.SymLogNorm):
        return mc.SymLogNorm(norm.linthresh, linscale=norm.linscale, vmin=norm.vmin, vmax=norm.vmax, base=norm.base)
    return mc.Normalize(vmin=norm.vmin, vmax=norm.vmax)


def _proxy(h):
    """A legend proxy made without an axes (qtplot.Line2D or qtplot.Patch) as its matplotlib artist, else None."""
    if isinstance(h, q.Line2D) and h.axes is None:
        from matplotlib.lines import Line2D
        return Line2D([0], [0], **_line_kw(h._style, h.get_label(), h.zorder))
    if isinstance(h, q.Patch):
        from matplotlib.patches import Patch
        return Patch(color=h.get_facecolor(), label=h.get_label() if not str(h.get_label()).startswith('_') else None)
    return None


def _ls(ls):
    """A matplotlib line style with every spelling of 'no line' (None, 'none', 'None', '', ' ') as 'none'."""
    return 'none' if ls in (None, 'none', 'None', '', ' ') else ls


def _px_to_pt(px):
    """Logical pixels (at qtplot.DPI) to points."""
    return float(px) / q.pt2px(1.0)


def _axis_hidden_ticks(axis):
    """Whether a qtplot tick axis was given no ticks at all (set_xticks([]): pyqtgraph's _tickLevels holds only empty levels)."""
    lev = getattr(axis, '_tickLevels', None)
    return lev is not None and all(len(level) == 0 for level in lev)


# ---------------------------------------------------------------------- #
#  The rebuild                                                             #
# ---------------------------------------------------------------------- #

def to_matplotlib(fig):
    """
    The qtplot figure as a matplotlib Figure (no GUI backend: Figure + savefig), at the window's
    design size, white, holding every visible axes at its figure fraction (_axes), the colour bars
    (_colorbar), and the figure texts of the plot area.  A window arranged by its Qt layout
    (automatic layout: the one rule for every window) is laid out at its design size for the
    rebuild (Figure.at_design_size), so the file does not depend on the window's size on screen;
    every spatial panel keeps its x:y ratio and every row its one height there as on screen.
    """
    with fig.at_design_size():
        return _to_matplotlib(fig)


def _to_matplotlib(fig):
    """to_matplotlib's rebuild, at the figure's current layout."""
    from matplotlib.figure import Figure
    W0, H0 = fig._design_px
    mfig = Figure(figsize=(W0 / q.DPI, H0 / q.DPI), dpi=q.DPI)
    mfig.patch.set_facecolor(fig.patch.get_facecolor() if hasattr(fig, 'patch') else 'white')
    axes = [ax for ax in fig._axes if ax._visible]
    cbar_axes = {id(ax) for ax in axes if getattr(ax, '_colorbar', None) is not None}
    made = {}
    for ax in axes:
        if id(ax) in cbar_axes:
            continue
        made[id(ax)] = _axes(mfig, ax)
    for ax in axes:
        if id(ax) in cbar_axes:
            _colorbar(mfig, ax)
    for ax in axes:                      # shared y: the second axes shows no y values (as on screen)
        if ax._shared_y is not None and id(ax) in made:
            made[id(ax)].tick_params(axis='y', labelleft=False)
    # Figure texts of the plot area only: a figure text below the lowest panel (by more than 2 % of the
    # height) sits in the control strip, a window's readout line beside its controls, and is left out.
    bottom = min((_pos(ax)[1] for ax in axes), default=0.0)
    for t in fig._texts:
        if t.get_visible() and t.get_text() and t.figure_xy()[1] >= bottom - 0.02:
            _text(t, transform=mfig.transFigure, target=mfig)
    return mfig


def _pos(ax):
    """The axes' rectangle as shown (after the aspect, colour-bar clearance, ...), in figure fractions."""
    x0, y0, w, h = ax._px_rect()
    W, H = ax.figure._canvas_size()
    return [x0 / W, 1.0 - (y0 + h) / H, w / W, h / H]


def _axes(mfig, ax):
    """
    One qtplot axes as a matplotlib axes in `mfig`: its artists (_artist), the quad meshes drawn
    straight into its view, the limits as shown (inverted axes included), the aspect, the labels
    (short_label, as on screen), tick sizes and hidden ticks, the grid, frame and axis state, the
    title, a secondary y axis, and the legend as on screen.  Text sizes are the points set on the
    qtplot axes.
    """
    from matplotlib.transforms import Affine2D
    m = mfig.add_axes(_pos(ax))
    m.set_facecolor(ax.get_facecolor() if hasattr(ax, 'get_facecolor') else 'white')
    # A rotated axes (Axes.set_rotated) turns as on screen: every artist drawn through the swap (x, y)
    # -> (y, x) into the view's axes, whose limits (inverted x) complete the turn.
    swap = Affine2D(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])) if getattr(ax, '_rotated', False) else None
    handle_map = {}
    for a in ax._children:
        if not a.get_visible():
            continue
        made = _artist(m, a, swap)
        if made is not None:
            handle_map[id(a)] = made
    # Quad meshes added straight to the view (RowHeatmap, PixelMap): filled polygons.
    for it in getattr(ax.vb, 'addedItems', []):
        if isinstance(it, q.QuadItem) and it.isVisible() and not any(it is getattr(a, 'item', None) for a in ax._children):
            _turned(_quads(m, it), swap, m)
    # limits as shown
    (xa, xb), (ya, yb) = ax.vb.viewRange()
    if ax.vb.state['xInverted']:
        xa, xb = xb, xa
    if ax.vb.state['yInverted']:
        ya, yb = yb, ya
    if ax._aspect == 'equal':
        m.set_aspect('equal', adjustable='box' if getattr(ax, '_adjust_box', False) else 'datalim')
    m.set_xlim(xa, xb)
    m.set_ylim(ya, yb)
    # labels, ticks, title
    xi, yi = ax.xaxis_item, ax.yaxis_item
    xl, yl = (ax._ylabel, ax._xlabel) if swap is not None else (ax._xlabel, ax._ylabel)   # the labels as the tick axes show them
    if xl:
        m.set_xlabel(q.short_label(xl), fontsize=xi._label_pt)
    if yl:
        m.set_ylabel(q.short_label(yl), fontsize=yi._label_pt)
    m.tick_params(axis='x', labelsize=xi._tick_pt)
    m.tick_params(axis='y', labelsize=yi._tick_pt)
    if _axis_hidden_ticks(xi):
        m.set_xticks([])
    if _axis_hidden_ticks(yi):
        m.set_yticks([])
    if not xi.style['showValues']:
        m.tick_params(axis='x', labelbottom=False)
    if not yi.style['showValues']:
        m.tick_params(axis='y', labelleft=False)
    if not yi.style['tickLength']:
        m.tick_params(axis='y', left=False)
    if not xi.style['tickLength']:
        m.tick_params(axis='x', bottom=False)
    if xi.grid is not False or yi.grid is not False:
        m.grid(True, lw=0.4, alpha=0.4)
    if not ax._axison:
        m.axis('off')
    elif not ax._frame:
        m.set_frame_on(False)
    if ax._title is not None and ax._title.get_text():
        m.set_title(ax._title.get_text(), fontsize=ax._title._size)
    for side in ax._side_axes:
        mode = getattr(side, '_mode', None)
        if mode is not None and mode[0] == 'func':
            sec = m.secondary_yaxis('right', functions=(mode[1], mode[2]))
            if side.labelText:
                sec.set_ylabel(q.short_label(side.labelText), fontsize=side._label_pt)
    # legends as on screen: the current one and the ones kept beside it (add_artist)
    legends = []
    if ax._legend is not None and getattr(ax, '_legend_args', None) is not None:
        legends.append(ax._legend_args)
    legends.extend((a, k) for _, a, k in getattr(ax, '_extra_legends', []))
    for k, (args, kw) in enumerate(legends):
        loc = kw.get('loc', 'best')
        loc = 'upper right' if loc in ('best', 0, 1) else loc
        lk = dict(loc=loc, frameon=kw.get('frameon', True), fontsize=q.font_pt(kw.get('fontsize'), 10.0),
                  ncol=kw.get('ncol', 1))
        if kw.get('title'):
            lk['title'] = kw['title']
        handles = kw.get('handles') if kw.get('handles') is not None else (args[0] if len(args) >= 2 else None)
        labels = kw.get('labels') if kw.get('labels') is not None else (args[1] if len(args) >= 2 else None)
        leg = None
        if handles is not None:
            hs = [handle_map.get(id(h)) or _proxy(h) for h in handles]
            pairs = [(h, l) for h, l in zip(hs, labels) if h is not None and l and not str(l).startswith('_')]
            if pairs:
                leg = m.legend([p[0] for p in pairs], [p[1] for p in pairs], **lk)
        else:
            leg = m.legend(**lk)
        if leg is not None and k < len(legends) - 1:
            m.add_artist(leg)               # kept when the next legend is made
    return m


def _line_kw(st, label, zorder):
    """
    matplotlib's plot keywords for a qtplot Line2D style dict (colour, width, style, alpha,
    markers); a label starting with '_' is dropped, as matplotlib's legend drops it.
    """
    kw = dict(color=st['color'], lw=st['lw'], ls=_ls(st['ls']), alpha=st['alpha'], zorder=zorder,
              label=label if (label and not str(label).startswith('_')) else None)
    if st.get('marker'):
        kw.update(marker=st['marker'], ms=st['ms'], mew=st['mew'])
        if st.get('mec') is not None:
            kw['mec'] = st['mec']
        if st.get('mfc') is not None:
            kw['mfc'] = st['mfc']
    return kw


def _turned(art, swap, m):
    """`art` (an artist, a contour set or None) drawn through `swap` (a rotated axes; None: as it is): its transform composed with the data transform (a scatter's offsets); returns `art`."""
    if art is None or swap is None:
        return art
    parts = getattr(art, 'collections', None) if not hasattr(art, 'set_transform') else None
    for c in (parts if parts is not None else [art]):
        if hasattr(c, 'set_offset_transform') and hasattr(c, 'get_offsets'):
            c.set_offset_transform(swap + m.transData)
        else:
            c.set_transform(swap + m.transData)
    return art


def _artist(m, a, swap=None):
    """
    One qtplot artist `a` drawn into the matplotlib axes `m` (through `swap`, the turn of a
    rotated axes, when given: _turned; a reference line or span turns the other way; a data
    text's position is swapped), returning the matplotlib artist (the
    legend's handle) or None when nothing is drawn: Line2D -> plot; _RefLine -> axhline / axvline;
    _Span -> axvspan / axhspan; AxesImage -> imshow with its extent, origin, colour map and norm,
    interpolation 'none' (the raw pixels), and its Affine2D (a rotated and translated topography)
    composed with the data transform; QuadMesh -> pcolormesh (NaN cells masked); ContourSet ->
    contourf from the arguments it was made with; LineCollection and PolyCollection -> the
    matching collections; Annotation -> annotate; Text -> text (empty texts skipped).  Anything
    else is not exported.
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.transforms import Affine2D
    turned = swap is not None
    if isinstance(a, q.Line2D):
        ln, = m.plot(a._x, a._y, **_line_kw(a._style, a.get_label(), a.zorder))
        return _turned(ln, swap, m)
    if isinstance(a, q._RefLine):
        st = a._style
        f = m.axhline if a._horizontal != turned else m.axvline
        return f(a._value, color=st['color'], lw=st['lw'], ls=_ls(st['ls']), alpha=st['alpha'], zorder=a.zorder,
                 label=a.get_label() if not str(a.get_label()).startswith('_') else None)
    if isinstance(a, q._Span):
        lo, hi, vertical, color, alpha = a._span
        f = m.axvspan if vertical != turned else m.axhspan
        return f(lo, hi, color=color, alpha=alpha, lw=0, zorder=a.zorder,
                 label=a.get_label() if not str(a.get_label()).startswith('_') else None)
    if isinstance(a, q.AxesImage):
        if not a.item.isVisible():
            return None
        im = m.imshow(a._A, cmap=_mpl_cmap(a.cmap), norm=_mpl_norm(a.norm), origin=a.origin, extent=a._extent,
                      interpolation='none', aspect='auto', zorder=a.zorder)
        if a._affine is not None or turned:
            tr = Affine2D(a._affine.get_matrix()) if a._affine is not None else Affine2D()
            im.set_transform((tr + swap if turned else tr) + m.transData)
        return im
    if isinstance(a, q.QuadMesh):
        return _turned(m.pcolormesh(a._X, a._Y, np.ma.masked_invalid(a._C), cmap=_mpl_cmap(a.cmap), norm=_mpl_norm(a.norm),
                                    shading='flat', zorder=a.zorder), swap, m)
    if isinstance(a, q.ContourSet):
        X, Y, Z, levels, cmap, norm = a._args
        return _turned(m.contourf(X, Y, Z, levels=levels, cmap=_mpl_cmap(cmap), norm=_mpl_norm(norm), zorder=a.zorder,
                                  alpha=a.get_alpha() if hasattr(a, 'get_alpha') else None), swap, m)
    if isinstance(a, q.TriContourSet):
        x, y, z, levels, cmap, norm, offsets = a._args
        out = None
        if len(x) >= 3:
            for dx, dy in offsets:
                try:
                    out = _turned(m.tricontourf(x + dx, y + dy, z, levels=levels, cmap=_mpl_cmap(cmap), norm=_mpl_norm(norm),
                                                zorder=a.zorder, alpha=a.get_alpha()), swap, m)
                except Exception as exc:          # a point set matplotlib cannot triangulate either
                    print(f"export: tricontourf skipped ({exc})")
        return out
    if isinstance(a, q.PathCollection):
        xy = a.get_offsets()
        if not len(xy):
            return None
        rows = a.get_facecolors()
        n = len(xy)
        sizes = np.resize(a.get_sizes(), (n,))
        ec = a._ec
        ekw = {}
        if ec is None or (isinstance(ec, str) and ec in ('none', 'None')):
            ekw['edgecolors'] = 'none'
        elif isinstance(ec, str) and ec == 'face':
            ekw['edgecolors'] = 'face'
        else:
            ekw['edgecolors'] = q._rgba_rows(ec, n, a._alpha)
        lw = float(np.ravel(a._lw)[0]) if np.ndim(a._lw) else float(a._lw)
        return _turned(m.scatter(xy[:, 0], xy[:, 1], s=sizes, c=rows, marker=a._marker if isinstance(a._marker, str) else 'o',
                                 linewidths=lw, zorder=a.zorder,
                                 label=a.get_label() if not str(a.get_label()).startswith('_') else None, **ekw), swap, m)
    if isinstance(a, q.LineCollection):
        cols = np.asarray(a._seg_colors(), dtype=float) / 255.0
        lws = np.resize(a._lw, (len(a._segments),)) if len(a._segments) else a._lw
        lc = LineCollection(a._segments, colors=cols, linewidths=lws, linestyles=_ls(a._ls) if a._ls != '-' else 'solid',
                            zorder=a.zorder)
        m.add_collection(lc)
        return _turned(lc, swap, m)
    if isinstance(a, q.PolyCollection):
        if a.item is None:
            return None
        return _turned(_quads(m, a.item, zorder=a.zorder), swap, m)
    if isinstance(a, q.Annotation):
        p = a._ann
        kw = dict(color=p['color'], fontsize=q.font_pt(p['fontsize'], 10.0), fontweight=p['fontweight'],
                  ha=p['ha'], va=p['va'], zorder=a.zorder)
        if p['arrowprops'] is not None:
            kw['arrowprops'] = p['arrowprops']
        return m.annotate(p['s'] or '', xy=p['xy'], xytext=p['xytext'], textcoords=p['textcoords'], **kw)
    if isinstance(a, q.Text):
        if not a.get_text():
            return None
        tr = m.transAxes if a._coords == 'axes' else None
        return _text(a, transform=tr, target=m, swap_xy=turned and a._coords == 'data')
    return None


def _quads(m, item, zorder=None):
    """
    A qtplot QuadItem (the item of a RowHeatmap, PixelMap or PolyCollection) as a matplotlib
    PolyCollection of its kept quads with their RGBA colours (uint8 -> 0..1); each cell edged by a
    hairline of its own colour so that a vector viewer shows no seams between cells.  None when the
    item holds no quad.  `zorder` overrides the item's z value.
    """
    from matplotlib.collections import PolyCollection
    quads = getattr(item, '_quads', None)
    cols = getattr(item, '_colors', None)
    if quads is None or cols is None or not len(quads):
        return None
    pc = PolyCollection(quads, facecolors=np.asarray(cols, dtype=float) / 255.0, edgecolors='face', linewidths=0.1,
                        antialiased=False, zorder=item.zValue() if zorder is None else zorder)
    m.add_collection(pc)
    return pc


def _text(t, transform=None, target=None, swap_xy=False):
    """
    A qtplot Text as matplotlib text on `target` (a matplotlib Axes or Figure): size in points,
    alignment, colour, rotation, weight, a monospace family, the bbox keys matplotlib takes,
    `transform` (axes or figure coordinates; None for data), and the qtplot pixel offset as an
    offset in points.
    """
    kw = dict(fontsize=t._size, ha=t._ha, va=t._va, color=t._color, rotation=t._rotation)
    if t._weight:
        kw['fontweight'] = t._weight
    if t._family in ('monospace', 'mono', 'courier'):
        kw['family'] = 'monospace'
    if t._bbox:
        b = dict(t._bbox)
        kw['bbox'] = {k: v for k, v in b.items() if k in ('facecolor', 'fc', 'alpha', 'edgecolor', 'ec', 'lw',
                                                         'linewidth', 'pad', 'boxstyle')}
    if transform is not None:
        kw['transform'] = transform
    if t._offset_px != (0.0, 0.0):
        from matplotlib.transforms import offset_copy
        base = transform if transform is not None else target.transData
        kw['transform'] = offset_copy(base, fig=target.figure if hasattr(target, 'figure') else target,
                                      x=_px_to_pt(t._offset_px[0]), y=_px_to_pt(t._offset_px[1]), units='points')
    x, y = t.figure_xy() if getattr(t, '_cell', None) is not None else (t._x, t._y)
    if swap_xy:
        x, y = y, x                            # a data text on a rotated axes sits at view (y, x)
    return target.text(x, y, t.get_text(), **kw)


def _colorbar(mfig, cax):
    """
    The colour bar held by the qtplot axes `cax` (cax._colorbar) as a matplotlib colour bar in an
    axes at the same position: the mappable's colour map and norm (so a gamma spaces the ticks as
    on screen), the label (short_label) and the tick size in points.
    """
    from matplotlib.cm import ScalarMappable
    cb = cax._colorbar
    m = mfig.add_axes(_pos(cax))
    sm = ScalarMappable(norm=_mpl_norm(getattr(cb.mappable, 'norm', None)), cmap=_mpl_cmap(cb.mappable.cmap))
    mcb = mfig.colorbar(sm, cax=m)
    if cb._label:
        mcb.set_label(q.short_label(cb._label), fontsize=cb._axis._label_pt)
    m.tick_params(labelsize=cb._axis._tick_pt)
    return mcb


# ---------------------------------------------------------------------- #
#  Saving                                                                  #
# ---------------------------------------------------------------------- #

def export(fig, path, dpi=300, facecolor=None, transparent=False):
    """
    Save `fig` through matplotlib: the format from the extension (pdf, eps, ps, svg, png, tif,
    jpg); text stays text in the vector formats (TrueType fonts embedded in PDF and PS / EPS, text
    elements in SVG; editable in Illustrator); raster formats at `dpi`.  The file is cropped to
    what is drawn, with a 0.05 inch margin.  The background is `facecolor` when given, else the
    figure's patch (fig.patch: white unless a window set it; a transparent patch, or
    `transparent`, gives a transparent background).  Returns `path`.
    """
    import matplotlib
    mfig = to_matplotlib(fig)
    if facecolor is None:
        facecolor = fig.patch.get_facecolor() if hasattr(fig, 'patch') else 'white'
    with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        # Cropped to what is drawn (the window's control strip is empty in the figure).
        mfig.savefig(path, dpi=dpi, facecolor=facecolor, transparent=bool(transparent), bbox_inches='tight',
                     pad_inches=0.05)
    return path
