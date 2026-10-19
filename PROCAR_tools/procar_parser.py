# -*- coding: utf-8 -*-
"""
procar_parser.py -- the single PROCAR reader of the PROCAR_tools package.

Built from complex_procar_parser.py (ComplexProcarParser) and generalized so
one parser covers every PROCAR VASP writes:

  * ISPIN=1, ISPIN=2 (second projection set starting at a second
    'k-point 1' line) and noncollinear / SOC (total block followed by the
    three magnetization blocks);
  * LORBIT=10 (s p d columns), 11 (lm columns), 12/13/14 (lm columns plus
    the complex phase block); the orbital column set is read from the
    file's own 'ion ...' header, so spd (9), spdf (16) or any other set
    works without changes here;
  * any number of k-points, bands and ions; single-ion files without the
    'tot' summary rows are handled because navigation is by the declared
    ion count, never by landmark rows.

Detection is structural (header and count driven). INCAR, when present, is
read only as a cross-check whose disagreement is reported, never trusted
over the file. A layout the rules below do not recognize raises ValueError
with the offending line number instead of returning zeros.

Companion files read from the same directory:
  DOSCAR  Fermi energy (line 6, token 4); optional, fermi_level=None if absent
  POSCAR  ion index (1-based) -> element symbol map; optional
  INCAR   LSORBIT / LNONCOLLINEAR / LORBIT cross-check; optional

Author: Benjamin Kafin
"""

import os
import re
from collections import defaultdict

import numpy as np


_RE_COUNTS = re.compile(
    r"#\s*of\s*k-points:\s*(\d+)\s*#\s*of\s*bands:\s*(\d+)\s*#\s*of\s*ions:\s*(\d+)",
    re.IGNORECASE)
_RE_KPOINT = re.compile(r"^\s*k[- ]?point\s+(\d+)\s*:(.*?)weight\s*=\s*([-+\d.Ee]+)",
                        re.IGNORECASE)
_RE_KPOINT_ANY = re.compile(r"^\s*k[- ]?point\s+(\d+)\b", re.IGNORECASE)
_RE_BAND = re.compile(
    r"^\s*band\s+(\d+)\s*#\s*energy\s+([-+\d.Ee]+)\s*#\s*occ\.\s+([-+\d.Ee]+)",
    re.IGNORECASE)
_RE_FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")
_RE_INT_TOKEN = re.compile(r"^\d+$")


def orbital_groups(orbital_names):
    """Map the file's orbital column names to l groups.

    Returns dict {'s': [cols], 'p': [cols], 'd': [cols], 'f': [cols]} of
    0-based column indices. Works for the LORBIT=10 names (s, p, d), the lm
    names (s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2, and the f columns VASP
    writes for spdf sets). A name that fits no group is left out of all of
    them.
    """
    groups = {'s': [], 'p': [], 'd': [], 'f': []}
    for i, name in enumerate(orbital_names):
        n = name.lower()
        if n == 's':
            groups['s'].append(i)
        elif n.startswith('p'):
            groups['p'].append(i)
        elif n.startswith('d') or n.startswith('x2'):
            groups['d'].append(i)
        elif n.startswith('f'):
            groups['f'].append(i)
    return groups


class ProcarParser:
    """
    Parse one directory's PROCAR (plus DOSCAR/POSCAR/INCAR when present).

    After parse() the arrays below are filled (spin axis always present,
    length ispin):

      kpoints      (nkpts, 3)      fractional k-point coordinates
      kweights     (nkpts,)        k-point weights
      energies     (ispin, nkpts, nbands)   raw eigenvalues (eV)
      occupations  (ispin, nkpts, nbands)
      total        (ispin, nkpts, nbands, nions, norb)   magnitude block
      ion_tot      (ispin, nkpts, nbands, nions)         per-ion 'tot' column
      tot_rows     (ispin, nkpts, nbands, norb + 1)      file 'tot' rows, NaN if absent
      mag          (3, nkpts, nbands, nions, norb)       NCL magnetization blocks, else None
      mag_ion_tot  (3, nkpts, nbands, nions)             NCL per-ion 'tot' columns, else None
      mag_tot_rows (3, nkpts, nbands, norb + 1)          NCL 'tot' rows, else None
      phase        (ispin, nkpts, nbands, nions, norb)   complex coefficients, else None
      phase_ion_sum(ispin, nkpts, nbands, nions)         trailing per-ion column of the
                                                         LORBIT=14 phase rows, NaN if absent
      charge_rows  (ispin, nkpts, nbands, norb + 1)      phase-block 'charge' rows, NaN if absent

    plus the scalars nkpts, nbands, nions, norb, ispin, ncl, has_phase,
    has_tot_rows, orbital_names, title, fermi_level, atom_type_map, and the
    INCAR cross-check dict incar_info.

    self.states (list of dicts, one per (spin, k-point, band), raw energies)
    is built by parse() for the tools that iterate states; get_states()
    returns the same dicts with energies shifted to the Fermi level on request.
    """

    def __init__(self, directory=".", filter_types=None, lsorbit=None):
        self.directory = directory
        self.procar_path = os.path.join(directory, "PROCAR")
        self.poscar_path = os.path.join(directory, "POSCAR")
        self.doscar_path = os.path.join(directory, "DOSCAR")
        self.incar_path = os.path.join(directory, "INCAR")
        self.filter_types = filter_types or []
        self.lsorbit_hint = lsorbit          # explicit override of the INCAR hint, cross-check only

        self.fermi_level = self.get_fermi_level_from_doscar(directory)
        self.atom_type_map = self.parse_poscar()
        self.incar_info = self._read_incar_hints()

        # Filled by parse()
        self.title = ""
        self.nkpts = self.nbands = self.nions = self.norb = 0
        self.ispin = 1
        self.ncl = False
        self.has_phase = False
        self.has_tot_rows = True
        self.orbital_names = []
        self.kpoints = self.kweights = None
        self.energies = self.occupations = None
        self.total = self.ion_tot = self.tot_rows = None
        self.mag = self.mag_ion_tot = self.mag_tot_rows = None
        self.phase = self.phase_ion_sum = self.charge_rows = None
        self.states = []            # one dict per (spin, k-point, band), raw energies
        self.states_combined = []

    # ------------------------------------------------------------------
    # Companion files
    # ------------------------------------------------------------------
    def get_fermi_level_from_doscar(self, directory):
        """Fermi energy from DOSCAR (line 6, token 4). None if DOSCAR is absent."""
        doscar_path = os.path.join(directory, "DOSCAR")
        if not os.path.exists(doscar_path):
            print(f"[procar_parser] DOSCAR not found in {directory}; fermi_level = None")
            return None
        with open(doscar_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 6:
            raise ValueError(f"{doscar_path}: fewer than 6 lines; cannot read the Fermi energy.")
        return float(lines[5].split()[3])

    def parse_poscar(self):
        """Map ion index (1-based) -> element symbol from POSCAR (VASP5+)."""
        mapping = {}
        if not os.path.exists(self.poscar_path):
            print(f"[procar_parser] POSCAR not found in {self.directory}; atom_type_map is empty")
            return mapping
        with open(self.poscar_path) as f:
            lines = [ln.strip() for ln in f]
        if len(lines) < 7:
            return mapping
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        idx = 1
        for elem, num in zip(elements, counts):
            for _ in range(num):
                mapping[idx] = elem
                idx += 1
        return mapping

    def _read_incar_hints(self):
        """LSORBIT / LNONCOLLINEAR / LORBIT from INCAR, for the cross-check only."""
        info = {"lsorbit": None, "lnoncollinear": None, "lorbit": None}
        if not os.path.isfile(self.incar_path):
            return info
        try:
            with open(self.incar_path, "r") as f:
                text = f.read().upper()
        except OSError:
            return info
        text = re.sub(r"[!#].*", "", text)
        for tag, key in (("LSORBIT", "lsorbit"), ("LNONCOLLINEAR", "lnoncollinear")):
            m = re.search(rf"{tag}\s*=\s*\.?\s*([TF])", text)
            if m:
                info[key] = (m.group(1) == "T")
        m = re.search(r"LORBIT\s*=\s*(\d+)", text)
        if m:
            info["lorbit"] = int(m.group(1))
        return info

    # ------------------------------------------------------------------
    # PROCAR
    # ------------------------------------------------------------------
    @staticmethod
    def _next_nonblank(lines, i):
        n = len(lines)
        while i < n and not lines[i].strip():
            i += 1
        return i

    @staticmethod
    def _first_token(line):
        t = line.split()
        return t[0] if t else ""

    def parse(self):
        """Read the PROCAR into the arrays described in the class docstring."""
        if not os.path.exists(self.procar_path):
            raise FileNotFoundError(f"PROCAR not found: {self.procar_path}")
        with open(self.procar_path, "r") as f:
            lines = f.read().splitlines()
        if len(lines) < 8:
            raise ValueError(f"{self.procar_path}: too short to be a PROCAR ({len(lines)} lines).")

        self.title = lines[0].strip()
        title_phase = "phase" in self.title.lower()

        m = _RE_COUNTS.search(lines[1])
        if not m:
            raise ValueError(f"{self.procar_path}: line 2 is not the counts line: {lines[1]!r}")
        self.nkpts, self.nbands, self.nions = (int(g) for g in m.groups())
        nkpts, nbands, nions = self.nkpts, self.nbands, self.nions

        # --- locate every k-point line; a second 'k-point 1' starts spin 2 ---
        kpt_lines = []            # (line index, kpoint index)
        for i, ln in enumerate(lines):
            mk = _RE_KPOINT_ANY.match(ln)
            if mk:
                kpt_lines.append((i, int(mk.group(1))))
        if not kpt_lines:
            raise ValueError(f"{self.procar_path}: no 'k-point' lines found.")
        starts = [pos for pos, (i, k) in enumerate(kpt_lines) if k == 1]
        if len(starts) not in (1, 2):
            raise ValueError(f"{self.procar_path}: found {len(starts)} 'k-point 1' lines; "
                             "expected 1 (ISPIN=1 / NCL) or 2 (ISPIN=2).")
        self.ispin = len(starts)
        ispin = self.ispin
        sections = []
        for s in range(ispin):
            lo = starts[s]
            hi = starts[s + 1] if s + 1 < ispin else len(kpt_lines)
            sections.append(kpt_lines[lo:hi])
        for s, sec in enumerate(sections):
            if [k for _, k in sec] != list(range(1, nkpts + 1)):
                raise ValueError(f"{self.procar_path}: spin section {s + 1} has k-point indices "
                                 f"{[k for _, k in sec]}; expected 1..{nkpts}.")

        # --- allocate ---
        self.kpoints = np.full((nkpts, 3), np.nan)
        self.kweights = np.full(nkpts, np.nan)
        self.energies = np.full((ispin, nkpts, nbands), np.nan)
        self.occupations = np.full((ispin, nkpts, nbands), np.nan)
        total = ion_tot = tot_rows = None
        mag = mag_ion_tot = mag_tot_rows = None
        phase = phase_ion_sum = charge_rows = None
        n_mag_blocks = None          # 1 (collinear) or 4 (NCL), fixed by the first band
        has_phase = None             # fixed by the first band
        tot_rows_seen = None

        for s, sec in enumerate(sections):
            for pos, (i_k, k_index) in enumerate(sec):
                k = k_index - 1
                # end of this k-point block = next k-point line (any spin) or EOF
                next_k_line = len(lines)
                for (j, _) in kpt_lines:
                    if j > i_k:
                        next_k_line = j
                        break
                mk = _RE_KPOINT.match(lines[i_k])
                if not mk:
                    raise ValueError(f"{self.procar_path}: line {i_k + 1}: k-point line without "
                                     f"a weight: {lines[i_k]!r}")
                coords = _RE_FLOAT.findall(mk.group(2))
                if len(coords) != 3:
                    raise ValueError(f"{self.procar_path}: line {i_k + 1}: expected 3 k-point "
                                     f"coordinates, found {len(coords)}: {lines[i_k]!r}")
                kp = np.array([float(c) for c in coords])
                kw = float(mk.group(3))
                if s == 0:
                    self.kpoints[k] = kp
                    self.kweights[k] = kw
                elif not (np.allclose(self.kpoints[k], kp) and np.isclose(self.kweights[k], kw)):
                    raise ValueError(f"{self.procar_path}: line {i_k + 1}: spin-2 k-point {k_index} "
                                     "differs from the spin-1 k-point of the same index.")

                i = i_k + 1
                bands_seen = 0
                while True:
                    i = self._next_nonblank(lines, i)
                    if i >= next_k_line:
                        break
                    mb = _RE_BAND.match(lines[i])
                    if not mb:
                        # tolerated: a repeated counts line (ISPIN=2 files) or any
                        # non-band line between k-point blocks
                        if _RE_COUNTS.search(lines[i]) or self._first_token(lines[i]).lower() in ("#",):
                            i += 1
                            continue
                        raise ValueError(f"{self.procar_path}: line {i + 1}: expected a band line, "
                                         f"got {lines[i]!r}")
                    b_index = int(mb.group(1))
                    if b_index != bands_seen + 1:
                        raise ValueError(f"{self.procar_path}: line {i + 1}: band {b_index} where "
                                         f"band {bands_seen + 1} was expected.")
                    if b_index > nbands:
                        raise ValueError(f"{self.procar_path}: line {i + 1}: band {b_index} exceeds "
                                         f"the declared {nbands} bands.")
                    b = b_index - 1
                    self.energies[s, k, b] = float(mb.group(2))
                    self.occupations[s, k, b] = float(mb.group(3))
                    i += 1

                    # ---- orbital header of the first magnitude block ----
                    i = self._next_nonblank(lines, i)
                    hdr = lines[i].split() if i < len(lines) else []
                    if not hdr or hdr[0].lower() != "ion" or hdr[-1].lower() != "tot":
                        raise ValueError(f"{self.procar_path}: line {i + 1}: expected an "
                                         f"'ion ... tot' header, got {lines[i] if i < len(lines) else 'EOF'!r}")
                    names = hdr[1:-1]
                    if not self.orbital_names:
                        self.orbital_names = names
                        self.norb = len(names)
                        norb = self.norb
                        total = np.zeros((ispin, nkpts, nbands, nions, norb))
                        ion_tot = np.zeros((ispin, nkpts, nbands, nions))
                        tot_rows = np.full((ispin, nkpts, nbands, norb + 1), np.nan)
                    elif names != self.orbital_names:
                        raise ValueError(f"{self.procar_path}: line {i + 1}: orbital header changed "
                                         f"to {names} (first band had {self.orbital_names}).")
                    norb = self.norb
                    i += 1

                    # ---- magnitude blocks: nions rows (+ optional 'tot' row) each ----
                    blocks = []       # list of (vals (nions, norb), ion_tot (nions,), tot_row or None)
                    while True:
                        vals = np.zeros((nions, norb))
                        itot = np.zeros(nions)
                        for r in range(nions):
                            i = self._next_nonblank(lines, i)
                            tok = lines[i].split() if i < len(lines) else []
                            if not tok or not _RE_INT_TOKEN.match(tok[0]) or int(tok[0]) != r + 1:
                                raise ValueError(f"{self.procar_path}: line {i + 1}: expected ion row "
                                                 f"{r + 1}, got {lines[i] if i < len(lines) else 'EOF'!r}")
                            if len(tok) < norb + 2:
                                raise ValueError(f"{self.procar_path}: line {i + 1}: ion row has "
                                                 f"{len(tok) - 1} values; expected {norb + 1} "
                                                 f"({norb} orbitals + tot).")
                            row = [float(x) for x in tok[1:norb + 2]]
                            vals[r] = row[:norb]
                            itot[r] = row[norb]
                            i += 1
                        trow = None
                        j = self._next_nonblank(lines, i)
                        if j < len(lines) and self._first_token(lines[j]).lower() == "tot":
                            tt = lines[j].split()[1:]
                            if len(tt) < norb + 1:
                                raise ValueError(f"{self.procar_path}: line {j + 1}: 'tot' row has "
                                                 f"{len(tt)} values; expected {norb + 1}.")
                            trow = np.array([float(x) for x in tt[:norb + 1]])
                            i = j + 1
                        blocks.append((vals, itot, trow))
                        # another magnitude block follows if the next line is an ion row
                        j = self._next_nonblank(lines, i)
                        if j < len(lines) and j < next_k_line and _RE_INT_TOKEN.match(self._first_token(lines[j])):
                            continue
                        break

                    if n_mag_blocks is None:
                        n_mag_blocks = len(blocks)
                        if n_mag_blocks not in (1, 4):
                            raise ValueError(f"{self.procar_path}: band 1 has {n_mag_blocks} magnitude "
                                             "blocks; expected 1 (collinear) or 4 (noncollinear).")
                        self.ncl = (n_mag_blocks == 4)
                        if self.ncl:
                            mag = np.zeros((3, nkpts, nbands, nions, norb))
                            mag_ion_tot = np.zeros((3, nkpts, nbands, nions))
                            mag_tot_rows = np.full((3, nkpts, nbands, norb + 1), np.nan)
                        tot_rows_seen = blocks[0][2] is not None
                        self.has_tot_rows = tot_rows_seen
                    elif len(blocks) != n_mag_blocks:
                        raise ValueError(f"{self.procar_path}: band {b_index} (k-point {k_index}, "
                                         f"spin {s + 1}) has {len(blocks)} magnitude blocks; the "
                                         f"first band had {n_mag_blocks}.")
                    if (blocks[0][2] is not None) != tot_rows_seen:
                        raise ValueError(f"{self.procar_path}: band {b_index}: 'tot' row presence "
                                         "differs from the first band.")
                    if self.ncl and ispin == 2:
                        raise ValueError(f"{self.procar_path}: noncollinear blocks together with two "
                                         "spin sections; not a layout VASP writes.")

                    total[s, k, b] = blocks[0][0]
                    ion_tot[s, k, b] = blocks[0][1]
                    if blocks[0][2] is not None:
                        tot_rows[s, k, b] = blocks[0][2]
                    if self.ncl:
                        for c in range(3):
                            mag[c, k, b] = blocks[c + 1][0]
                            mag_ion_tot[c, k, b] = blocks[c + 1][1]
                            if blocks[c + 1][2] is not None:
                                mag_tot_rows[c, k, b] = blocks[c + 1][2]

                    # ---- optional phase block: 'ion' header without 'tot' ----
                    j = self._next_nonblank(lines, i)
                    phase_here = False
                    if j < len(lines) and j < next_k_line:
                        ph = lines[j].split()
                        if ph and ph[0].lower() == "ion":
                            if ph[-1].lower() == "tot":
                                raise ValueError(f"{self.procar_path}: line {j + 1}: a fifth "
                                                 "'ion ... tot' header; layout not recognized.")
                            if ph[1:] != self.orbital_names:
                                raise ValueError(f"{self.procar_path}: line {j + 1}: phase header "
                                                 f"orbitals {ph[1:]} differ from {self.orbital_names}.")
                            phase_here = True
                            i = j + 1
                    if has_phase is None:
                        has_phase = phase_here
                        self.has_phase = phase_here
                        if phase_here:
                            phase = np.zeros((ispin, nkpts, nbands, nions, norb), dtype=complex)
                            phase_ion_sum = np.full((ispin, nkpts, nbands, nions), np.nan)
                            charge_rows = np.full((ispin, nkpts, nbands, norb + 1), np.nan)
                    elif phase_here != has_phase:
                        raise ValueError(f"{self.procar_path}: band {b_index} (k-point {k_index}, "
                                         f"spin {s + 1}): phase block presence differs from band 1.")
                    if phase_here:
                        for r in range(nions):
                            i = self._next_nonblank(lines, i)
                            tok = lines[i].split() if i < len(lines) else []
                            if not tok or not _RE_INT_TOKEN.match(tok[0]) or int(tok[0]) != r + 1:
                                raise ValueError(f"{self.procar_path}: line {i + 1}: expected phase "
                                                 f"row {r + 1}, got {lines[i] if i < len(lines) else 'EOF'!r}")
                            if len(tok) < 1 + 2 * norb:
                                raise ValueError(f"{self.procar_path}: line {i + 1}: phase row has "
                                                 f"{len(tok) - 1} values; expected {2 * norb} "
                                                 "(Re, Im per orbital).")
                            vals = [float(x) for x in tok[1:1 + 2 * norb]]
                            phase[s, k, b, r] = np.array(vals[0::2]) + 1j * np.array(vals[1::2])
                            if len(tok) >= 2 + 2 * norb:
                                phase_ion_sum[s, k, b, r] = float(tok[1 + 2 * norb])
                            i += 1
                        j = self._next_nonblank(lines, i)
                        if j < len(lines) and self._first_token(lines[j]).lower() == "charge":
                            cv = lines[j].split()[1:]
                            charge_rows[s, k, b, :min(len(cv), norb + 1)] = [float(x) for x in cv[:norb + 1]]
                            i = j + 1
                    bands_seen += 1
                if bands_seen != nbands:
                    raise ValueError(f"{self.procar_path}: k-point {k_index} (spin {s + 1}) has "
                                     f"{bands_seen} bands; the header declares {nbands}.")

        self.total, self.ion_tot, self.tot_rows = total, ion_tot, tot_rows
        self.mag, self.mag_ion_tot, self.mag_tot_rows = mag, mag_ion_tot, mag_tot_rows
        self.phase, self.phase_ion_sum, self.charge_rows = phase, phase_ion_sum, charge_rows

        self._cross_check(title_phase)
        self.states = self._build_states(fermi_relative=False)
        print(f"[procar_parser] {self.procar_path}: {self.nkpts} k-points, {self.nbands} bands, "
              f"{self.nions} ions, {self.norb} orbitals {self.orbital_names}; ISPIN={self.ispin}, "
              f"noncollinear={self.ncl}, phase block={self.has_phase}, tot rows={self.has_tot_rows}; "
              f"{len(self.states)} states.")
        return self

    def _cross_check(self, title_phase):
        """Report (never enforce) disagreements between the file structure and the
        title line / INCAR / explicit lsorbit hint."""
        if title_phase != self.has_phase:
            print(f"[procar_parser] NOTE: title line {self.title!r} "
                  f"{'mentions' if title_phase else 'does not mention'} 'phase' but the file "
                  f"{'has' if self.has_phase else 'has no'} phase block; using the file structure.")
        hint = self.lsorbit_hint
        if hint is None:
            inc = self.incar_info
            if inc["lsorbit"] is not None or inc["lnoncollinear"] is not None:
                hint = bool(inc["lsorbit"]) or bool(inc["lnoncollinear"])
        if hint is not None and hint != self.ncl:
            print(f"[procar_parser] NOTE: INCAR/hint says noncollinear={hint} but the PROCAR "
                  f"structure says noncollinear={self.ncl}; using the file structure.")
        lorbit = self.incar_info.get("lorbit")
        if lorbit is not None:
            expect_phase = lorbit in (2, 12, 13, 14)
            if expect_phase != self.has_phase:
                print(f"[procar_parser] NOTE: INCAR LORBIT={lorbit} but the PROCAR "
                      f"{'has' if self.has_phase else 'has no'} phase block; using the file structure.")

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def _build_states(self, fermi_relative=False):
        if self.total is None:
            raise RuntimeError("parse() must run before states are requested.")
        shift = 0.0
        if fermi_relative:
            if self.fermi_level is None:
                raise ValueError("fermi_relative=True needs the Fermi energy, but DOSCAR was "
                                 f"not found in {self.directory}.")
            shift = self.fermi_level
        headers = list(self.orbital_names) + ["tot"]
        out = []
        for s in range(self.ispin):
            for k in range(self.nkpts):
                for b in range(self.nbands):
                    mat = np.concatenate([self.total[s, k, b], self.ion_tot[s, k, b][:, None]], axis=1)
                    st = {
                        "spin": s + 1,
                        "kpoint": k + 1,
                        "kweight": float(self.kweights[k]),
                        "band": b + 1,
                        "band_name": f"band {b + 1}",
                        "energy": float(self.energies[s, k, b]) - shift,
                        "occ": float(self.occupations[s, k, b]),
                        "orbital_headers": headers,
                        "contrib_by_ion": {r + 1: dict(zip(headers, mat[r])) for r in range(self.nions)},
                    }
                    if self.has_tot_rows:
                        trow = self.tot_rows[s, k, b]
                        st["orbital_matrix"] = np.vstack([mat, trow])
                        st["orbital_totals"] = dict(zip(headers, trow))
                    else:
                        st["orbital_matrix"] = mat
                    if self.ncl:
                        st["mag"] = self.mag[:, k, b]
                    if self.has_phase:
                        st["phase_headers"] = list(self.orbital_names)
                        st["coeff_matrix"] = self.phase[s, k, b]
                    out.append(st)
        return out

    def get_states(self, fermi_relative=False):
        """State dicts, one per (spin, k-point, band). fermi_relative=True shifts
        'energy' by the DOSCAR Fermi energy; False returns self.states (raw
        eigenvalues)."""
        if not fermi_relative:
            return self.states if self.states else self._build_states(False)
        return self._build_states(True)

    def states_by_index(self):
        """{(spin_idx, k_idx, band_idx) (all 0-based): (nions, norb) total magnitudes}."""
        if self.total is None:
            raise RuntimeError("parse() must run before states are requested.")
        out = {}
        for s in range(self.ispin):
            for k in range(self.nkpts):
                for b in range(self.nbands):
                    out[(s, k, b)] = self.total[s, k, b]
        return out

    # ------------------------------------------------------------------
    # Carried over from ComplexProcarParser (phase-block tools)
    # ------------------------------------------------------------------
    def _require_phase(self, what):
        if not self.has_phase:
            raise ValueError(f"{what} needs the complex phase block (LORBIT=12/13/14); "
                             f"{self.procar_path} has none.")

    def build_state_types(self):
        """Tag each state with element types that have nonzero contribution."""
        self._require_phase("build_state_types")
        for st in self.states:
            types = set()
            C = st["coeff_matrix"]
            mag2 = np.abs(C) ** 2
            for ion_idx, elem in self.atom_type_map.items():
                if mag2[ion_idx - 1].sum() > 0:
                    types.add(elem)
            st["types"] = sorted(types)

    def filter_states(self):
        """Remove states whose contributing types are a subset of filter_types."""
        if not self.filter_types:
            return
        ft = {x.upper() for x in self.filter_types}
        kept = []
        for st in self.states:
            types = {t.upper() for t in st.get("types", [])}
            if types and types.issubset(ft):
                continue
            kept.append(st)
        self.states = kept
        print(f"After filtering, {len(self.states)} states remain.")

    def combine_kpoint_bands(self):
        """
        Combine k-points coherently for each (spin, band):
            C_comb = (Sum_k w_k * C_k) / (Sum_k w_k)
        Energies in the combined states are relative to the Fermi level when
        DOSCAR was found, raw otherwise.
        """
        self._require_phase("combine_kpoint_bands")
        shift = self.fermi_level if self.fermi_level is not None else 0.0
        groups = defaultdict(list)
        for st in self.states:
            groups[(st["spin"], st["band"])].append(st)

        comb = []
        for (spin, band), sts in groups.items():
            W = sum(s["kweight"] * 1.0 for s in sts)
            C_sum = sum(s["coeff_matrix"] * s["kweight"] for s in sts)
            Cc = C_sum / W
            E_avg = sum((s["energy"] - shift) * s["kweight"] for s in sts) / W
            occ = sum(s["occ"] * s["kweight"] for s in sts) / W

            comb.append({
                "spin": spin,
                "band": band,
                "energy": E_avg,
                "occ": occ,
                "coeff_matrix": Cc,
                "phase_headers": sts[0]["phase_headers"],
                "total_weight": W,
            })

        self.states_combined = sorted(comb, key=lambda x: (x["spin"], x["energy"]))
        return self.states_combined

    def export_combined_bands(self, out_filename="combined_complex_composition.txt"):
        """Write per-band complex-composition table (per (spin, band) for ISPIN=2)."""
        self.combine_kpoint_bands()
        fpath = os.path.join(self.directory, out_filename)
        with open(fpath, "w") as fout:
            for idx, st in enumerate(self.states_combined, start=1):
                hdrs = st["phase_headers"]
                Cc = st["coeff_matrix"]
                N, M = Cc.shape
                spin_tag = f"\tSpin: {st['spin']}" if self.ispin == 2 else ""
                fout.write(
                    f"Band {idx}{spin_tag}\tOriginalBand: {st['band']}"
                    f"\tEnergy: {st['energy']:.6f} eV\n"
                )
                headers = ["Ion"]
                for orb in hdrs:
                    headers += [f"{orb}_Re", f"{orb}_Im"]
                fout.write("\t".join(headers) + "\n")
                for ion_idx in sorted(self.atom_type_map):
                    if ion_idx <= N:
                        row = Cc[ion_idx - 1]
                    else:
                        row = np.zeros(M, dtype=complex)
                    fields = [str(ion_idx)]
                    for c in row:
                        fields += [f"{c.real:.4e}", f"{c.imag:.4e}"]
                    fout.write("\t".join(fields) + "\n")
                fout.write("\n")
        print(f"Exported combined complex bands to {fpath}")

    def run(self):
        self.parse()
        self.build_state_types()
        self.filter_states()
        self.export_combined_bands()


if __name__ == "__main__":
    parser = ProcarParser(".", [])
    parser.run()
