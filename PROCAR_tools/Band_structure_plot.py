# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:45:00 2026
@author: Benjamin Kafin

Educational Version: Integrated VASP Fatband Plotter.
This version is fully generic, mapping any provided 'filter_types'
to RGB channels without hard-coded element names.

Spin-mode detection added: a single DOSCAR read now determines the Fermi
energy AND auto-detects the spin configuration (collinear unpolarized,
collinear polarized, or non-collinear / SOC) from the site-projected
column count. The detected mode is stored on self.spin_mode for later use.

BAND.dat blocks are canonicalized to ascending k (VASPKIT writes zigzag
order), aligning every band row-for-row with PROCAR/EIGENVAL k-point order.
Under NCL/SOC, bands degenerate along the entire k-path (Kramers doublets)
are merged into one scatter group each -- mean energy, member-mean
projections combined before normalization -- with a combined hover
annotation. Collinear modes draw exactly as before.

PROCAR reading goes through the package's shared ProcarParser (procar_parser.py),
so ISPIN=1/2, noncollinear, LORBIT=10/11/12/14 and any orbital column set
(s/p/d, 9 lm, 16 lm with f) are handled in one place. Element mode sums every
orbital column of an element; orbital mode keeps its s/p/d RGB channels.
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
from collections import defaultdict
from enum import Enum, auto
from procar_parser import ProcarParser, orbital_groups

try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False


class SpinMode(Enum):
    COLLINEAR_UNPOL = auto()
    COLLINEAR_POL = auto()
    NCL_SOC = auto()


# Column-count -> SpinMode mapping (unambiguous by VASP format)
#   ISPIN=1:  3 (l-decomposed), 9 (lm-decomposed spd), 16 (lm spdf)
#   ISPIN=2:  6, 18, 32
#   NCL/SOC:  12 (l x 4), 36 (lm-spd x 4), 64 (lm-spdf x 4)
_COL_TO_MODE = {}
for _n in (3, 9, 16):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_UNPOL
for _n in (6, 18, 32):
    _COL_TO_MODE[_n] = SpinMode.COLLINEAR_POL
for _n in (12, 36, 64):
    _COL_TO_MODE[_n] = SpinMode.NCL_SOC


# Orbital column groups (s/p/d/f) are derived from the PROCAR header by
# procar_parser.orbital_groups, so the lm column order and count come from the file.
# Orbital -> RGB channel for orbital-mode hue (s=Red, p=Green, d=Blue), mirroring
# the element-mode 0=R,1=G,2=B convention.
_ORBITAL_ORDER = ['s', 'p', 'd']


class CPUOrbitalBandPlotter:
    """
    Handles ingestion of VASP output files and generates an interactive
    band structure plot using generic N-element normalization.
    """
    def __init__(self, directory=".", filter_types=None, color_by='element',
                 soc_pair_thresh=0.001):
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.band_dat_path = os.path.join(directory, "BAND.dat")
        self.klabels_path = os.path.join(directory, "KLABELS")
        self.filter_types = filter_types or []
        self.color_by = color_by   # 'element' (default) or 'orbital'
        # eV; whole-path degeneracy tolerance for Kramers grouping (NCL/SOC only)
        self.soc_pair_thresh = soc_pair_thresh
        self.spin_mode = None
        self.fermi_level = self.parse_doscar(directory)
        self.atom_type_map = self.parse_poscar()

        print(f"--- Initialization Complete ---")
        print(f"Fermi Level (from DOSCAR): {self.fermi_level:.4f} eV")
        print(f"Detected Spin Mode: {self.spin_mode.name if self.spin_mode is not None else 'UNKNOWN'}")
        print(f"POSCAR Mapping: {len(self.atom_type_map)} ions found")
        print(f"Filters Active: {self.filter_types}")

        self.states = {} # O(1) Lookup: (spin, k_idx, b_idx)
        self.ispin = 1
        self.parser = None          # shared ProcarParser (set by parse_procar)
        self.norb = 0
        self.orbital_names = []

    def parse_doscar(self, directory):
        """Single DOSCAR read: returns the Fermi energy and sets self.spin_mode
        from the site-projected column count.

        Line 1 token 0 = ion count; line 6 (index 5) holds nedos (token 2) and
        the Fermi energy (token 3). Block 0 is the total DOS; the first
        site-projected block (block 1, preceded by a header line) gives the
        column count used to detect the spin configuration."""
        doscar_path = os.path.join(directory, "DOSCAR")
        if not os.path.exists(doscar_path):
            return 0.0
        with open(doscar_path, "r") as f:
            atomnum = int(f.readline().split()[0])
            for _ in range(4):
                f.readline()
            header = f.readline().split()
            nedos, ef = int(header[2]), float(header[3])

            # Block 0: total DOS (skip its nedos rows).
            for _ in range(nedos):
                f.readline()

            # Block 1: first site-projected atom block, preceded by a header
            # line. Read one data row to count its site-projected columns.
            if atomnum >= 1:
                f.readline()  # atom block header line
                first_row = f.readline().split()
                num_cols = len(first_row) - 1  # drop the leading energy column
                self.spin_mode = _COL_TO_MODE.get(num_cols)

        return ef

    def parse_poscar(self):
        mapping = {}
        if not os.path.exists(self.poscar_path): return mapping
        with open(self.poscar_path) as f:
            lines = [ln.strip() for ln in f]
        elements, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        idx = 1
        for elem, num in zip(elements, counts):
            for _ in range(num):
                mapping[idx] = elem
                idx += 1
        return mapping

    def parse_procar(self):
        """Reads PROCAR through the package's shared ProcarParser (procar_parser.py).

        Stores self.states[(s_idx, k_idx, b_idx)] -> ndarray (N_ions, norb) of the
        total-charge projected magnitudes, with norb and the column names taken
        from the file (s/p/d for LORBIT=10, 9 lm columns for spd, 16 for spdf).
        ISPIN=2 gives s_idx 0 and 1; noncollinear files contribute their total
        block (the magnetization blocks stay available on self.parser.mag). The
        DOSCAR-detected spin mode is compared with the PROCAR structure and a
        disagreement is printed; the PROCAR structure is used for the states."""
        if not os.path.exists(self.procar_path): return
        print(f"\n>>> Starting PROCAR Parsing (shared procar_parser)...")

        self.parser = ProcarParser(self.directory)
        self.parser.parse()
        self.ispin = self.parser.ispin
        self.norb = self.parser.norb
        self.orbital_names = list(self.parser.orbital_names)
        self.states = self.parser.states_by_index()

        if self.parser.ncl:
            procar_mode = SpinMode.NCL_SOC
        elif self.ispin == 2:
            procar_mode = SpinMode.COLLINEAR_POL
        else:
            procar_mode = SpinMode.COLLINEAR_UNPOL
        if self.spin_mode is not None and self.spin_mode != procar_mode:
            print(f"NOTE: DOSCAR spin mode {self.spin_mode.name} differs from the PROCAR "
                  f"structure {procar_mode.name}; the PROCAR structure is used for the states.")

        print(f"<<< Finished Parsing. Locked {len(self.states)} states (N_ions={self.parser.nions}, "
              f"{self.norb} orbital columns {self.orbital_names}).")

    def parse_band_dat(self):
        band_data = []
        if not os.path.exists(self.band_dat_path): return []
        with open(self.band_dat_path, 'r') as f:
            current_k, current_e = [], []
            for line in f:
                if line.startswith('# Band-Index:'):
                    if current_k: band_data.append({'k': np.array(current_k), 'e': np.array(current_e)})
                    current_k, current_e = [], []
                    continue
                if not line.startswith('#') and line.strip():
                    p = line.split()
                    current_k.append(float(p[0]))
                    current_e.append(float(p[1]))
            if current_k: band_data.append({'k': np.array(current_k), 'e': np.array(current_e)})
        # VASPKIT writes BAND.dat in zigzag order (every other block runs
        # k-descending so line plots connect without pen lifts). Canonicalize
        # to ascending k so every band aligns row-for-row with PROCAR/EIGENVAL
        # k-point order -- otherwise state-tensor colors and cursor k indices
        # come out mirrored on the descending bands.
        for band in band_data:
            if len(band['k']) >= 2 and band['k'][-1] < band['k'][0]:
                band['k'] = band['k'][::-1].copy()
                band['e'] = band['e'][::-1].copy()
        return band_data

    def _detect_soc_pairs(self, thresh):
        """Group bands that are degenerate along the ENTIRE k-path (Kramers
        doublets under SOC). Walks bands in index order; band b+1 joins band
        b's group iff their canonicalized BAND.dat energy arrays agree within
        thresh at every shared k-point (chains > 2 allowed; a band that pairs
        with nothing becomes a singleton group). Whole-path agreement means
        isolated crossings never merge distinct doublets.

        Returns a list of {'members': [0-based band indices], 'dE': per-k
        max-min spread across members (None for singletons)}."""
        groups = []
        n_bands = len(self.band_raw)
        b = 0
        while b < n_bands:
            members = [b]
            while b + 1 < n_bands:
                e_a, e_b = self.band_raw[b]['e'], self.band_raw[b + 1]['e']
                nk = min(len(e_a), len(e_b))
                if nk == 0 or np.abs(e_a[:nk] - e_b[:nk]).max() > thresh:
                    break
                members.append(b + 1)
                b += 1
            if len(members) > 1:
                nk = min(len(self.band_raw[m]['e']) for m in members)
                stack = np.stack([self.band_raw[m]['e'][:nk] for m in members])
                dE = stack.max(axis=0) - stack.min(axis=0)
            else:
                dE = None
            groups.append({'members': members, 'dE': dE})
            b += 1
        return groups

    def parse_klabels(self):
        """Parse the VASPKIT KLABELS file into (tick_coords, tick_labels).

        Logic matches the working unfold_bands.py reader: skip empty lines and
        any line containing 'K-Label' or '*'; take the first token as the label
        and the second as the coordinate; convert GAMMA -> $\\Gamma$. If the file
        is absent or unreadable, returns empty lists so the plot keeps normal
        numeric x-axis ticks."""
        tick_coords, tick_labels = [], []
        if not os.path.exists(self.klabels_path):
            return tick_coords, tick_labels
        with open(self.klabels_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or 'K-Label' in line or '*' in line:
                    continue
                try:
                    label = parts[0]
                    coord = float(parts[1])
                    if label.upper() == 'GAMMA':
                        label = r'$\Gamma$'
                    tick_labels.append(label)
                    tick_coords.append(coord)
                except (ValueError, IndexError):
                    continue
        return tick_coords, tick_labels

    def _apply_klabels(self, ax):
        """Apply KLABELS high-symmetry ticks + grey guide lines to an axis.
        Absent/empty KLABELS leaves the default numeric x-axis untouched."""
        tick_coords, tick_labels = self.parse_klabels()
        if tick_coords:
            for coord in tick_coords:
                ax.axvline(x=coord, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xticks(tick_coords)
            ax.set_xticklabels(tick_labels)

    def _build_state_tensor(self):
        """Stack self.states into ONE dense array, ONCE, so both color modes are
        fast array operations and toggling is instant.

        Builds self._state_arr with shape (n_spin, n_bands, max_k, N_ions, norb),
        indexed to match the draw loops: [s_idx, b_idx, k_idx, ion, lm]. Lookup
        misses stay zero. Ragged bands are padded to max_k; per-band valid k-count
        is recorded in self._band_nk so draws only plot real points.

        This is the single heavy step; everything downstream (element sums,
        orbital s/p/d, per-element population, both modes, every toggle) reads
        from this tensor with vectorized numpy, never re-walking the dict."""
        band_raw = self.band_raw
        n_bands = len(band_raw)
        self._band_nk = [len(b['k']) for b in band_raw]
        max_k = max(self._band_nk, default=0)

        # Infer N_ions from any stored state (all share the same first dim).
        n_ions = 0
        for v in self.states.values():
            n_ions = v.shape[0]
            break

        arr = np.zeros((self.ispin, n_bands, max_k, n_ions, self.norb), dtype=np.float32)
        for (s_idx, k_idx, b_idx), mg in self.states.items():
            if s_idx < self.ispin and b_idx < n_bands and k_idx < max_k:
                arr[s_idx, b_idx, k_idx] = mg
        self._state_arr = arr
        self._n_ions = n_ions

        # Column-group index arrays for s/p/d, from the file's own orbital names
        # (f columns, when present, are summed in element mode only).
        groups = orbital_groups(self.orbital_names)
        self._s_cols = np.array(groups['s'], dtype=int)
        self._p_cols = np.array(groups['p'], dtype=int)
        self._d_cols = np.array(groups['d'], dtype=int)

        # Map each element to its 0-based ion indices.
        self._elem_ion0 = {}
        for ion_i, elem in self.atom_type_map.items():
            self._elem_ion0.setdefault(elem, []).append(ion_i - 1)

    def _elem_spd(self, elem):
        """Vectorized per-element s/p/d magnitudes from the state tensor.
        Returns array (n_spin, n_bands, max_k, 3) = (s_sum, p_sum, d_sum) summed
        over that element's ions. Pure array ops; no Python per-point loop."""
        ions0 = self._elem_ion0.get(elem, [])
        if not ions0:
            sh = self._state_arr.shape
            return np.zeros((sh[0], sh[1], sh[2], 3), dtype=np.float64)
        sub = self._state_arr[:, :, :, ions0, :]            # (S,B,K,n_e_ions,norb)
        s = sub[..., self._s_cols].sum(axis=(3, 4))         # (S,B,K)
        p = sub[..., self._p_cols].sum(axis=(3, 4))
        d = sub[..., self._d_cols].sum(axis=(3, 4))
        return np.stack([s, p, d], axis=-1)                 # (S,B,K,3)

    def _elem_total(self, elem):
        """Total population of an element: every orbital column (s, p, d and f
        when present) summed over that element's ions. Returns (n_spin, n_bands, max_k)."""
        ions0 = self._elem_ion0.get(elem, [])
        if not ions0:
            sh = self._state_arr.shape
            return np.zeros((sh[0], sh[1], sh[2]), dtype=np.float64)
        return self._state_arr[:, :, :, ions0, :].sum(axis=(3, 4))

    def plot_colored_bands(self):
        """Build the interactive figure with an 'Orbital' toggle (bottom left).

        Toggle OFF (default): single-panel element mode, element->RGB using the
        input filter_types (None slots skip a channel).
        Toggle ON: one subplot per element NAMED IN filter_types (None slots
        skipped); coloring switches to orbital (hue = s/p/d blend, visibility =
        total population). Both modes read one precomputed state tensor, so
        clicking the toggle is instant."""
        self.band_raw = self.parse_band_dat()
        self._build_state_tensor()   # single heavy step; both modes read this
        self._soc_groups = None
        if self.spin_mode is SpinMode.NCL_SOC:
            self._soc_groups = self._detect_soc_pairs(self.soc_pair_thresh)
            multi = [g for g in self._soc_groups if len(g['members']) > 1]
            worst = max((g['dE'].max() for g in multi), default=0.0)
            print(f"[SOC] Kramers grouping at {self.soc_pair_thresh:g} eV: "
                  f"{len(self._soc_groups)} groups ({len(multi)} degenerate, "
                  f"{len(self._soc_groups) - len(multi)} singleton); "
                  f"worst in-group |dE| = {worst:.2e} eV")
        self.fig = plt.figure(figsize=(12, 8))
        # Axes -> mplcursors.Cursor registry (MOPmatcher pattern). The click
        # handler captures this dict by reference, so rebuilds that clear and
        # repopulate it are picked up without rewiring the event.
        self._cursor_by_axes = {}
        # 'Orbital' toggle, bottom left.
        self._ax_toggle = self.fig.add_axes([0.01, 0.01, 0.12, 0.06])
        self._chk_orbital = CheckButtons(self._ax_toggle, ['Orbital'],
                                         [getattr(self, 'color_by', 'element') == 'orbital'])
        self._chk_orbital.on_clicked(self._on_toggle)
        self._rebuild_figure()
        self._wire_global_click(self.fig)
        self._report_cursor_status()
        plt.show()

    def _report_cursor_status(self):
        """Console diagnostics for the interactive cursors: both failure modes
        (mplcursors missing from the running env, non-interactive backend) are
        otherwise silent."""
        backend = matplotlib.get_backend()
        if not HAS_MPLCURSORS:
            print("[cursor] mplcursors NOT importable in this Python environment "
                  "-- band/k-point cursors disabled. Install with: pip install mplcursors")
            return
        print(f"[cursor] mplcursors OK, backend={backend}, "
              f"cursors on {len(self._cursor_by_axes)} axes. "
              "Left-click a band panel to enable; right-click to clear.")
        if 'inline' in backend.lower() or backend.lower() == 'agg':
            print("[cursor] WARNING: this backend renders a STATIC image -- clicks and "
                  "hover never reach the plot. In Spyder set Preferences > IPython "
                  "console > Graphics > Backend to 'Qt5' or 'Automatic' (or run "
                  "'%matplotlib qt' in the console), then re-run.")

    def _on_toggle(self, label):
        states = self._chk_orbital.get_status()
        self.color_by = 'orbital' if states[0] else 'element'
        self._rebuild_figure()
        self.fig.canvas.draw_idle()

    def _rebuild_figure(self):
        """Clear and redraw the figure body for the current mode, preserving the
        toggle widget axes."""
        self._cursor_by_axes.clear()
        for ax in list(self.fig.axes):
            if ax is not self._ax_toggle:
                self.fig.delaxes(ax)
        if getattr(self, 'color_by', 'element') == 'orbital':
            self._draw_orbital_mode()
        else:
            self._draw_element_mode()

    def _attach_cursor(self, ax, artists, hover_map):
        """Attach an initially-disabled hover cursor to an axis (MOPmatcher
        pattern): the cursor exists but shows nothing until a left click on
        the axis arms it (see _wire_global_click). Targets the explicit
        scatter list so reference lines (Fermi line, KLABELS guides) never
        annotate. hover_map: scatter artist -> 1-based band number
        (collinear), or (tuple of 1-based member band numbers, per-k |dE|
        array or None) for NCL/SOC groups; the k-point comes from the
        selected point's index within the artist."""
        if not (HAS_MPLCURSORS and artists):
            return
        cur = mplcursors.cursor(artists, hover=True); cur.enabled = False
        self._cursor_by_axes[ax] = cur

        @cur.connect("add")
        def _on_add(sel, hmap=hover_map):
            info = hmap.get(sel.artist)
            if info is None:
                return
            # mplcursors >= 0.5 exposes sel.index; older releases use
            # sel.target.index. Scatter (PathCollection) indices are ints.
            idx = getattr(sel, "index", None)
            if idx is None:
                idx = getattr(sel.target, "index", None)
            if idx is None:
                return
            idx = int(idx)
            k_no = idx + 1
            if not isinstance(info, tuple):     # collinear: single band number
                sel.annotation.set_text(f"band {info}\nk-point {k_no}")
                return
            members, dE = info                  # NCL/SOC group
            if len(members) == 1:               # singleton: keep single-band text
                sel.annotation.set_text(f"band {members[0]}\nk-point {k_no}")
                return
            sel.annotation.set_text(
                "\n".join(f"band {b} k-point {k_no}" for b in members))

    def _wire_global_click(self, fig):
        """Figure-level click handler (MOPmatcher pattern): left-click arms
        the hover cursor of the axes under the mouse (annotations then follow
        the marker the mouse is over); right-click removes all annotations
        from that axis and disarms its cursor. Clicks outside the data axes
        (including the Orbital toggle) fall through the registry lookup."""
        if HAS_MPLCURSORS:
            def _on_click(event, _map=self._cursor_by_axes):
                if event.inaxes is None: return
                cur = _map.get(event.inaxes)
                if cur is None: return
                if event.button == 1:
                    if not cur.enabled:
                        cur.enabled = True
                        print("[cursor] enabled -- annotations follow the mouse; "
                              "right-click to clear")
                    # Annotate the marker under the click right away;
                    # mplcursors itself only picks on the next mouse motion.
                    try:
                        cur._on_select_event(event)
                    except Exception:
                        pass
                elif event.button == 3:
                    for sel in list(cur.selections):
                        cur.remove_selection(sel)
                    cur.enabled = False
                    print("[cursor] cleared and disabled")
            fig.canvas.mpl_connect("button_press_event", _on_click)

    def _draw_element_mode(self):
        """Single-panel coloring: each band point's RGB is the normalized
        composition of the named elements (0=R, 1=G, 2=B). None/falsy filter
        slots leave their channel empty. Works for any number of ions/elements,
        including one. Per-element weight is the total charge on that element
        (sum of all orbital columns over its ions), computed by vectorized array ops
        from the state tensor. One scatter call per band."""
        ax = self.fig.add_subplot(1, 1, 1)

        # Per-element total population: all orbital columns -> (S,B,K) for each filter element.
        active = [t for t in self.filter_types if t]
        # weights[channel] is (S,B,K); channel order follows filter_types (None -> 0).
        chan_weight = []
        for t in self.filter_types:
            if t:
                chan_weight.append(self._elem_total(t))  # (S,B,K) total charge on element
            else:
                chan_weight.append(None)

        # Stack the up-to-3 channels into an RGB weight array.
        S, B, K = self._state_arr.shape[0], self._state_arr.shape[1], self._state_arr.shape[2]
        rgbw = np.zeros((S, B, K, 3), dtype=np.float64)
        for ci in range(min(3, len(self.filter_types))):
            if chan_weight[ci] is not None:
                rgbw[..., ci] = chan_weight[ci]
        w_total = rgbw.sum(axis=-1)                # (S,B,K)
        with np.errstate(divide='ignore', invalid='ignore'):
            rgb = np.where(w_total[..., None] > 0, rgbw / w_total[..., None], 0.0)
        # Points with zero total -> black (matches original [0,0,0,1]).

        artists, hover_map = [], {}
        if self.spin_mode is SpinMode.NCL_SOC and getattr(self, '_soc_groups', None):
            # One scatter per whole-path degenerate group: y = member-mean
            # energy; channel weights are member-MEANED before normalization
            # (only the doublet sum/mean of the projections is gauge-invariant).
            for grp in self._soc_groups:
                ms = grp['members']
                nk = min(self._band_nk[m] for m in ms)
                k_coords = self.band_raw[ms[0]]['k'][:nk]
                e_coords = np.mean([self.band_raw[m]['e'][:nk] for m in ms], axis=0)
                gw = rgbw[0, ms, :nk, :].mean(axis=0)          # (nk, 3)
                gt = gw.sum(axis=-1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    g_rgb = np.where(gt[..., None] > 0, gw / gt[..., None], 0.0)
                colors = np.ones((nk, 4), dtype=np.float64)   # alpha 1
                colors[:, :3] = g_rgb
                sc = ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)
                artists.append(sc)
                hover_map[sc] = (tuple(m + 1 for m in ms), grp['dE'])
        else:
            for s_idx in range(self.ispin):
                for b_idx, band in enumerate(self.band_raw):
                    nk = self._band_nk[b_idx]
                    k_coords, e_coords = band['k'], band['e']
                    colors = np.ones((nk, 4), dtype=np.float64)   # alpha 1
                    colors[:, :3] = rgb[s_idx, b_idx, :nk, :]
                    sc = ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)
                    artists.append(sc)
                    hover_map[sc] = b_idx + 1
        self._attach_cursor(ax, artists, hover_map)

        base_colors = [[1,0,0], [0,1,0], [0,0,1]]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=e,
                          markerfacecolor=c, markersize=10)
                          for e, c in zip(self.filter_types, base_colors) if e]
        ax.legend(handles=legend_elements, title="Atomic Projections", loc='upper right')

        self._apply_klabels(ax)
        ax.axhline(0, color='red', ls='-', lw=0.8, alpha=0.5)
        ax.set_ylim(-3, 3); ax.set_ylabel('Energy (eV)')
        ax.set_title(f"Dynamic Normalized Fatbands: {self.filter_types}")

    def _draw_orbital_mode(self):
        """One subplot per element NAMED IN filter_types (None slots skipped).
        Within each element's panel, HUE = the s/p/d orbital blend (s=R, p=G,
        d=B, normalized like element mode), and VISIBILITY (alpha) = the element's
        total population at that point, self-scaled per element. Vectorized array
        ops from the state tensor; one scatter call per band."""
        # Elements to show = named (non-None) filter slots, de-duplicated, input order.
        elements = []
        for t in self.filter_types:
            if t and t not in elements:
                elements.append(t)
        if not elements:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No elements in filter_types to show.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        n_el = len(elements)
        for ax_idx, elem in enumerate(elements):
            ax = self.fig.add_subplot(1, n_el, ax_idx + 1)
            spd = self._elem_spd(elem)                # (S,B,K,3)
            pop = spd.sum(axis=-1)                     # (S,B,K) element total population
            max_pop = float(pop.max()) if pop.size else 0.0
            if max_pop <= 0:
                max_pop = 1e-30
            with np.errstate(divide='ignore', invalid='ignore'):
                hue = np.where(pop[..., None] > 0, spd / pop[..., None], 0.0)  # (S,B,K,3)
            alpha = np.minimum(1.0, pop / max_pop)    # (S,B,K)

            artists, hover_map = [], {}
            if self.spin_mode is SpinMode.NCL_SOC and getattr(self, '_soc_groups', None):
                # One scatter per whole-path degenerate group; s/p/d weights
                # member-MEANED before hue normalization (gauge-invariant), and
                # alpha on the same per-band max_pop scale as singletons.
                for grp in self._soc_groups:
                    ms = grp['members']
                    nk = min(self._band_nk[m] for m in ms)
                    k_coords = self.band_raw[ms[0]]['k'][:nk]
                    e_coords = np.mean([self.band_raw[m]['e'][:nk] for m in ms], axis=0)
                    g_spd = spd[0, ms, :nk, :].mean(axis=0)        # (nk, 3)
                    g_pop = g_spd.sum(axis=-1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        g_hue = np.where(g_pop[..., None] > 0, g_spd / g_pop[..., None], 0.0)
                    colors = np.empty((nk, 4), dtype=np.float64)
                    colors[:, :3] = g_hue
                    colors[:, 3] = np.minimum(1.0, g_pop / max_pop)
                    sc = ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)
                    artists.append(sc)
                    hover_map[sc] = (tuple(m + 1 for m in ms), grp['dE'])
            else:
                for s_idx in range(self.ispin):
                    for b_idx, band in enumerate(self.band_raw):
                        nk = self._band_nk[b_idx]
                        k_coords, e_coords = band['k'], band['e']
                        colors = np.empty((nk, 4), dtype=np.float64)
                        colors[:, :3] = hue[s_idx, b_idx, :nk, :]
                        colors[:, 3] = alpha[s_idx, b_idx, :nk]
                        # Points with zero population -> alpha 0 (invisible), matching original.
                        sc = ax.scatter(k_coords, e_coords, c=colors, s=15, edgecolors='none', zorder=2)
                        artists.append(sc)
                        hover_map[sc] = b_idx + 1
            self._attach_cursor(ax, artists, hover_map)

            orb_colors = [[1,0,0], [0,1,0], [0,0,1]]
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=orb,
                              markerfacecolor=c, markersize=10)
                              for orb, c in zip(_ORBITAL_ORDER, orb_colors)]
            ax.legend(handles=legend_elements, title="Orbitals", loc='upper right')

            self._apply_klabels(ax)
            ax.axhline(0, color='red', ls='-', lw=0.8, alpha=0.5)
            ax.set_ylim(-3, 3)
            ax.set_title(f"{elem}")
            if ax_idx == 0:
                ax.set_ylabel('Energy (eV)')

        self.fig.suptitle(f"Orbital-Resolved Fatbands (per element): {elements}", fontsize=13)

if __name__ == "__main__":
    # Element mode by default; click the bottom-left 'Orbital' toggle to switch
    # to per-element orbital subplots (only the elements named in filter_types).
    # SOC pair-merge test: Zach_bulkAu (NCL/SOC). For the collinear regression
    # run, switch to directory 'Zach_bulkSi' with filter_types=['Si'].
    plotter = CPUOrbitalBandPlotter(
        directory=r'dir/run',
        filter_types=['Si'])
    plotter.parse_procar()
    plotter.plot_colored_bands()