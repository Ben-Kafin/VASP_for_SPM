# -*- coding: utf-8 -*-
"""
Machinery: the spectrum decomposition (the decomp) of the simulator's window, everything in it that depends
on its level, called by every view of the simulator (Interactive_STM_Simulator in simulator.py).

The levels, and what a left click on a curve or a legend entry does at each:
    0  the markers: one curve per marker; the marker clicked becomes the active marker, level 1;
    1  the parts of the active marker's spectrum, Au and Molecule; level 2;
    2  its atom types, one curve each; the type clicked becomes the active element, level 3;
    3  the orbital types of the active element, each summed over the element's atoms; nothing (the last level).
A right click anywhere in the spectrum or the legend panel, or the panel's Back button, goes back one level
(levels 1 to 3).  The two ends do nothing: a right click at level 0, a left click at level 3.  A double click
has no action of its own.

Here: the state (Decomp: the level, the active marker, the active element); the partitions the maps and the
heatmaps show at each level (Decomp.partitions, with the level-3 orbital columns, Decomp.orbital_columns);
the spectrum's curves (Decomp.curves, Curve records in drawing order); the legend panel's title and entries
(Decomp.legend, Entry records; TITLES; all_labels, every label any level can show, for the panel's fixed
width); the click rules (Decomp.press, Decomp.down, Decomp.back); the curve a click is on (top_artist); the
hover labels of the orbital curves (hover_labels); the decomp's line in the info panel (Decomp.info).

The shared data stay in the simulator and are passed in: atomtypes and atomnums (the atom types and their
counts, in the POSCAR order), orbitals (the orbital names of the DOS columns), the atom types' colours, the
markers' colours in index order, dos_up_gpu (the per-atom DOS) and the displayed LDOS arrays.  Nothing here
draws, except the hover labels; the simulator plots the curves and finds the curve or entry under a click
(top_artist), and its one entry point for the clicks calls Decomp.press.  None of this is physics: it is the
display's bookkeeping.
"""

import colorsys
from collections import namedtuple

import numpy as np
import cupy as cp

from .qtgui import qtplot as plt
from .qtgui import widgets


# The legend panel's title at each level (level 0 has none).
TITLES = ('', 'Parts', 'Atoms', 'Orbitals')
# The last level: a left click there does nothing.
LAST_LEVEL = 3
# Level 1's parts: the gold atoms and every other atom.
PARTS = ('Au', 'Molecule')
# The faded-colour factor of the old per-atom curves (no curve uses it now).
S = 0.25


# One curve of the spectrum, in drawing order: its data y (over the displayed energies), its style (color,
# linestyle, linewidth, alpha, zorder; None: the plot's default), its label (the artist's label: 'marker_<i>'
# at level 0, the part, the atom type, '<element> <orbital>' at level 3; None for the faded totals), its
# picker and pickradius (None: not pickable), whether a left click on it goes down a level (clickable) with
# which option, the legend entry it stands for (an Entry, or None: the faded totals are not entries), and
# whether the hover labels sit on it (hover: level 3's orbital curves).
Curve = namedtuple('Curve', 'y color linestyle linewidth alpha zorder label picker pickradius '
                            'clickable option entry hover')

# One entry of the legend panel, in display order: its label ('' at level 0: the marker's colour only), its
# curve's colour, line style and line width (the sample), whether it is a clickable button (levels 0 to 2;
# level 3's entries are plain key rows) and the option a left click on it goes down with.
Entry = namedtuple('Entry', 'label color linestyle linewidth clickable option')


def _orbit_base(orb):
    """Machinery: an orbital name without its spin suffix ('_up' / '_down'): the name its line style goes by."""
    if orb.endswith('_up'):
        return orb[:-3]
    elif orb.endswith('_down'):
        return orb[:-5]
    return orb


def _column_orbitals(orbitals):
    """
    Machinery: the orbital names of the displayed LDOS columns, in column order, each once: every name of
    `orbitals` without its spin suffix (_orbit_base).  In a spin-polarised run `orbitals` has a name for each
    column of both spin channels ('s_up', 's_down', 'py_up', ...), while each channel's DOS (dos_up) has one
    column per orbital and the displayed LDOS combines the two channels column by column (the spin sum, or
    their difference under 'Mag': the simulator's _combine_ldos_display), so a column is named by its base
    orbital (s, py, pz, ...); without spin channels the names are the columns' own.
    """
    return list(dict.fromkeys(_orbit_base(o) for o in orbitals))


def _lighten_color(color, amount=0.3):
    """Machinery: a colour lightened in HLS by `amount` of the way to white (the old '_up' orbital curves; no curve uses it now)."""
    c = plt.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)


def _linestyle_map(orbitals):
    """Machinery: the orbital line styles, one per base orbital name over every orbital (sorted), its '_up' and '_down' columns alike."""
    unique_bases = sorted(set(_orbit_base(o) for o in orbitals))
    styles = ['-', '--', ':', '-.'] + [(0, (3 + i, 2)) for i in range(max(0, len(unique_bases) - 4))]
    return dict(zip(unique_bases, styles))


def _orbital_label(element, orb):
    """Machinery: the label of an element's orbital (level 3's partitions, curves and entries), e.g. 'C pz'."""
    return f"{element} {orb}"


def all_labels(atomtypes, orbitals):
    """
    Machinery: every label any level can show for the loaded data, in order, each once (the legend panel's
    fixed width is the widest of them): level 1's parts, level 2's atom types and level 3's
    '<element> <orbital>' for every atom type and every displayed column's orbital (_column_orbitals: its
    base orbital, no '_up' / '_down' names), whether or not its column is shown.  Level 0's entries have
    no label.
    """
    labels = list(PARTS) + list(atomtypes) + [_orbital_label(t, orb) for t in atomtypes
                                              for orb in _column_orbitals(orbitals)]
    return list(dict.fromkeys(labels))


def top_artist(artists):
    """
    Machinery: the artist on top among `artists` (given in drawing order): the highest zorder, and among equal
    zorders the one drawn last, as matplotlib and Qt stack them.  None when there are none.
    """
    top = None
    for a in artists:
        if top is None or a.get_zorder() >= top.get_zorder():
            top = a
    return top


def hover_labels(ax, lines, curves, old):
    """
    Machinery: the spectrum's hover labels after its curves were drawn anew: the old ones (`old`, a
    widgets.HoverLabels or None) removed, and new ones made only on the curves that carry them (level 3's
    orbital curves), on from the start; `lines` are the plotted curves, in the order of `curves`.  Returns
    the new widgets.HoverLabels, or None.
    """
    if old is not None:
        old.remove()
    hover = [ln for ln, c in zip(lines, curves) if c.hover]
    if not hover:
        return None
    return widgets.HoverLabels(ax, hover, enabled=True, arm_by_click=False)


class Decomp:
    """
    Machinery: the decomp's state and what follows from it: the level (0 to 3), the active marker (an index)
    and the active element (an atom type, or None).  The simulator reads and sets them as its plot_level,
    active_marker_idx and active_element; no value is checked.  The level-3 orbital columns are kept with the
    element and the DOS array they were found from (orbital_columns).
    """

    def __init__(self):
        """Machinery: level 0, marker 0, no element, no orbital columns found yet."""
        self.level = 0
        self.marker = 0
        self.element = None
        self._columns_key = None           # (element, the dos_up array object) the columns were found from
        self._columns = []

    # --- the level-3 orbital columns ---

    def orbital_columns(self, atomtypes, atomnums, orbitals, dos_up):
        """
        Machinery: the active element's orbital columns that are shown, [(column index, orbital name)]: every
        displayed column, named by its base orbital (_column_orbitals: one per column, the spin channels
        combined), except one where the element's DOS summed over its atoms (dos_up), and its first and second
        cp.gradient, all stay below 1e-5.  No columns without an element or without its atoms.
        The result is kept for the element and the dos_up array object (compared with `is`; the extra
        broadening makes a new array), so a drag computes nothing here.
        """
        key = self._columns_key
        if key is not None and key[0] == self.element and key[1] is dos_up:
            return self._columns
        cols = []
        if self.element is not None:
            atom_types_exp = np.repeat(atomtypes, atomnums)
            e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.element]
            for col_idx, orb in enumerate(_column_orbitals(orbitals)):
                if e_idx:
                    D = cp.sum(dos_up[e_idx, :, col_idx], axis=0)
                    dD = cp.gradient(D)
                    d2D = cp.gradient(dD)
                    if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                        cols.append((col_idx, orb))
        self._columns_key = (self.element, dos_up)
        self._columns = cols
        return cols

    # --- the maps' and heatmaps' partitions ---

    def partitions(self, f_ldos, atomtypes, atomnums, orbitals, dos_up):
        """
        Machinery: the partitions the maps and the heatmaps show at the level, [(label, data)], each data
        summed from f_ldos (points, energies, atoms, orbitals) over the atoms and orbitals it covers, an
        empty atom set giving zeros: level 0 'Total'; level 1 'Au' and 'Molecule' (every atom not Au); level
        2 each atom type; level 3 the active element's shown orbital columns (orbital_columns), each
        '<element> <orbital>'.  None for f_ldos gives none.
        """
        partitions = []
        if f_ldos is not None:
            if self.level == 0:
                partitions.append(("Total", cp.sum(f_ldos, axis=(2, 3))))
            elif self.level == 1:
                atom_types_exp = np.repeat(atomtypes, atomnums)
                au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
                mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
                au_data = cp.sum(f_ldos[:, :, au_idx, :], axis=(2, 3)) if au_idx else cp.zeros_like(f_ldos[:, :, 0, 0])
                mol_data = cp.sum(f_ldos[:, :, mol_idx, :], axis=(2, 3)) if mol_idx else cp.zeros_like(f_ldos[:, :, 0, 0])
                partitions.extend([(PARTS[0], au_data), (PARTS[1], mol_data)])
            elif self.level == 2:
                atom_types_exp = np.repeat(atomtypes, atomnums)
                for t in atomtypes:
                    t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                    t_data = cp.sum(f_ldos[:, :, t_idx, :], axis=(2, 3)) if t_idx else cp.zeros_like(f_ldos[:, :, 0, 0])
                    partitions.append((t, t_data))
            elif self.level == 3:
                atom_types_exp = np.repeat(atomtypes, atomnums)
                if self.element is not None:
                    e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.element]
                    for col_idx, orb in self.orbital_columns(atomtypes, atomnums, orbitals, dos_up):
                        orb_data = cp.sum(f_ldos[:, :, e_idx, col_idx], axis=2)
                        partitions.append((_orbital_label(self.element, orb), orb_data))
        return partitions

    # --- the spectrum's curves and the legend panel ---

    def curves(self, active, marker_spectra, marker_colors, atomtypes, atomnums, orbitals, dos_up, type_colors,
               energies, integral=None):
        """
        Machinery: the spectrum's curves at the level, [Curve] in drawing order.  `active`: the active marker's
        displayed spectrum (energies, atoms, orbitals); `marker_spectra`: every marker's, in index order (level
        0); `marker_colors`: the markers' colours in index order; `type_colors`: {atom type: colour} (grey
        for any other); `energies`: the displayed energies (an empty atom set gives zeros of their length);
        `integral`: the integral over the energies when 'Norm' is on (each curve of level 0 divided by its own
        integral, levels 1 to 3 divided by the active spectrum's total integral, + 1e-15), else None (also
        under 'Norm' when the spectra given are already divided: in a Line / Path drag the simulator divides
        them by its kept Norm factors).
          level 0: per marker, its total in its colour, lw 2.5, 'marker_<i>', pickable (5), an entry with no
                   label, clickable with the marker's index;
          level 1: the total faded (black, lw 1.5, alpha 0.1, zorder 1), then Au (orange) and Molecule (black),
                   lw 2, zorder 2, pickable (5), entries, clickable;
          level 2: per atom type, in its colour, lw 2, pickable (3), an entry, clickable with the type;
          level 3: the element's total faded (its colour, lw 2, alpha 0.15, zorder 1), then per shown orbital
                   column (orbital_columns: named by its base orbital, the spin channels combined in `active`)
                   the element's orbital: its colour, the orbital's line style, lw 1.2, zorder 10, not
                   pickable, an entry, not clickable, with the hover labels.
        """
        out = []
        if self.level == 0:
            for i, s in enumerate(marker_spectra):
                t_y = cp.sum(s.copy(), axis=(1, 2))
                if integral is not None:
                    t_y /= (integral(t_y) + 1e-15)
                color = marker_colors[i]
                entry = Entry('', color, '-', 2.5, True, i)
                out.append(Curve(t_y, color, '-', 2.5, None, None, f'marker_{i}', True, 5, True, i, entry, False))
            return out
        a = active.copy()
        if integral is not None:
            a /= (integral(cp.sum(a, axis=(1, 2))) + 1e-15)
        if self.level == 1:
            out.append(Curve(cp.sum(a, axis=(1, 2)), 'black', '-', 1.5, 0.1, 1, None, None, None, False, None, None,
                             False))
            atom_types_exp = np.repeat(atomtypes, atomnums)
            au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
            mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
            au_y = cp.sum(a[:, au_idx, :], axis=(1, 2)) if au_idx else np.zeros_like(energies)
            mol_y = cp.sum(a[:, mol_idx, :], axis=(1, 2)) if mol_idx else np.zeros_like(energies)
            for label, y, color in ((PARTS[0], au_y, 'orange'), (PARTS[1], mol_y, 'black')):
                entry = Entry(label, color, '-', 2, True, label)
                out.append(Curve(y, color, '-', 2, None, 2, label, True, 5, True, label, entry, False))
        elif self.level == 2:
            atom_types_exp = np.repeat(atomtypes, atomnums)
            for t in atomtypes:
                t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                t_y = cp.sum(a[:, t_idx, :], axis=(1, 2)) if t_idx else np.zeros_like(energies)
                color = type_colors.get(t, 'grey')
                entry = Entry(t, color, '-', 2, True, t)
                out.append(Curve(t_y, color, '-', 2, None, None, t, True, 3, True, t, entry, False))
        elif self.level == 3:
            atom_types_exp = np.repeat(atomtypes, atomnums)
            e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.element]
            color = type_colors.get(self.element, 'grey')
            e_y = cp.sum(a[:, e_idx, :], axis=(1, 2)) if e_idx else np.zeros_like(energies)
            out.append(Curve(e_y, color, '-', 2, 0.15, 1, None, None, None, False, None, None, False))
            linestyle_map = _linestyle_map(orbitals)
            for col_idx, orb in self.orbital_columns(atomtypes, atomnums, orbitals, dos_up):
                y_orb = cp.sum(a[:, e_idx, col_idx], axis=1)
                ls = linestyle_map[_orbit_base(orb)]
                label = _orbital_label(self.element, orb)
                entry = Entry(label, color, ls, 1.2, False, label)
                out.append(Curve(y_orb, color, ls, 1.2, None, 10, label, None, None, False, label, entry, True))
        return out

    def legend(self, curves):
        """
        Machinery: the legend panel for the curves drawn (curves()): (title, entries, back).  `title`: TITLES at
        the level ('' at level 0); `entries`: the curves' Entry records in drawing order (level 0: marker index
        order), without the faded totals (Map's energy markers are not curves); `back`: whether the Back button
        acts (levels 1 to 3).
        """
        return TITLES[self.level], [c.entry for c in curves if c.entry is not None], self.level > 0

    # --- the clicks ---

    def down(self, option):
        """
        Machinery: one level down with a curve's or an entry's option (at level 0 the option becomes the active
        marker, at level 2 the active element); True.  At the last level nothing changes; False.
        """
        if self.level >= LAST_LEVEL:
            return False
        if self.level == 0:
            self.marker = option
        elif self.level == 2:
            self.element = option
        self.level += 1
        return True

    def back(self):
        """Machinery: one level back (levels 1 to 3); True.  At level 0 nothing changes; False."""
        if self.level <= 0:
            return False
        self.level -= 1
        return True

    def press(self, button, dblclick, item):
        """
        Machinery: the click rules, one decision per click: `item` the Curve or Entry under the click (None on
        neither).  A double click's press does nothing (a double click is one left click); a right click goes
        back (back); a left click on a clickable item goes down with its option (down); anything else (a left
        click on neither or on a level-3 curve or entry, a middle click) does nothing.  True when the state
        changed (the caller redraws once), False when nothing changed (nothing is redrawn).
        """
        if dblclick:
            return False
        if button == plt.MouseButton.RIGHT:
            return self.back()
        if button == plt.MouseButton.LEFT and item is not None and item.clickable:
            return self.down(item.option)
        return False

    # --- the info panel ---

    def info(self):
        """
        Machinery: the decomp's line in the info panel: the level's full word and the selection: 'markers',
        'Partition (Au / Molecule), marker k', 'Atom Types, marker k', 'Orbitals of X, marker k'.
        """
        if self.level == 0:
            return "markers"
        if self.level == 1:
            what = "Partition (Au / Molecule)"
        elif self.level == 2:
            what = "Atom Types"
        else:
            what = f"Orbitals of {self.element}"
        return f"{what}, marker {self.marker}"
