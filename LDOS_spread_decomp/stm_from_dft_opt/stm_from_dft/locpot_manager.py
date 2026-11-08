# -*- coding: utf-8 -*-
"""
Auto-detecting LOCPOT handler.
Handles ISPIN=1, ISPIN=2, and non-collinear SOC LOCPOT files.

Author: Benjamin Kafin
"""

import numpy as np
from os.path import exists, join, getsize
from pymatgen.io.vasp import Locpot

from .doscar_parser import SpinMode


class LocpotManager:
    """
    Machinery: dedicated handler for VASP LOCPOT data extraction and caching.
    Auto-detects spin mode from the number of volumetric data sections.
    The cache, LOCPOT.npy, is written to and looked for at the path the
    caller gives (cache_path). The Simulator gives its one caches path,
    _cache_path('LOCPOT.npy') = <VASP folder>/caches/LOCPOT.npy, which also
    makes the folder; this class has no folder rule of its own. A LOCPOT.npy
    left directly in the VASP folder is neither moved nor read.
    """

    def __init__(self, filepath, spin_mode=None, *, cache_path):
        """
        Machinery: the handler for the LOCPOT in filepath, its cache at
        cache_path.

        Parameters
        ----------
        filepath : str
            Directory containing LOCPOT (the VASP folder).
        spin_mode : SpinMode or None
            If provided, used directly.  If None, auto-detected from LOCPOT
            section count.
        cache_path : str
            The path of the LOCPOT.npy cache, read and written as given (the
            Simulator's _cache_path('LOCPOT.npy'), whose folder it has made).
        """
        self.filepath = filepath
        self.locpot_path = join(filepath, "LOCPOT")
        # The cache at the caller's path (the Simulator's one caches path,
        # <VASP folder>/caches/LOCPOT.npy). An older LOCPOT.npy directly in
        # the VASP folder is not looked at, so it is neither moved nor read.
        self.cache_path = cache_path
        self.spin_mode = spin_mode
        self.data = None

    def get_data(self, force_rebuild=False):
        """
        Returns the LOCPOT data array.
        Checks for a valid cache before parsing raw VASP output.
        """
        if self._is_cache_valid() and not force_rebuild:
            print(f"[*] Valid LOCPOT cache detected at {self.cache_path}. Skipping parse.")
            self.data = np.load(self.cache_path)
            return self.data

        return self._rebuild_cache()

    def _is_cache_valid(self):
        """Machinery: validates existence and dimensionality of the .npy cache at cache_path."""
        if not exists(self.cache_path) or getsize(self.cache_path) == 0:
            return False
        try:
            cached_data = np.load(self.cache_path, mmap_mode="r")
            if self.spin_mode == SpinMode.COLLINEAR_POL:
                return cached_data.ndim == 4  # (2, X, Y, Z)
            else:
                # ISPIN=1 and NCL both store a single 3D grid
                return cached_data.ndim == 3
        except Exception:
            return False

    def _rebuild_cache(self):
        """
        Machinery: parses raw LOCPOT and applies spin-mode-appropriate extraction.
        Auto-detects spin mode from section count if not pre-set.
        Saves the result at cache_path (its folder made by the caller).
        """
        if not exists(self.locpot_path):
            raise FileNotFoundError(f"Source LOCPOT not found: {self.locpot_path}")

        print(f"[*] Parsing raw LOCPOT via pymatgen...")
        lpt = Locpot.from_file(self.locpot_path)

        # Key-blind sequential section extraction
        vol_sections = list(lpt.data.values())
        n_sections = len(vol_sections)

        # Auto-detect spin mode from section count if not provided
        if self.spin_mode is None:
            if n_sections >= 4:
                self.spin_mode = SpinMode.NCL_SOC
            elif n_sections == 2:
                self.spin_mode = SpinMode.COLLINEAR_POL
            else:
                self.spin_mode = SpinMode.COLLINEAR_UNPOL
            print(f"[*] Auto-detected LOCPOT spin mode: {self.spin_mode.name} ({n_sections} sections)")

        # Branch on spin mode
        if self.spin_mode == SpinMode.COLLINEAR_POL and n_sections >= 2:
            # Reconstruct spin channels: V_up = (V_tot + V_mag) / 2,
            #                            V_dn = (V_tot - V_mag) / 2
            v_tot = vol_sections[0]
            v_mag = vol_sections[1]
            self.data = np.stack([(v_tot + v_mag) / 2.0, (v_tot - v_mag) / 2.0])

        elif self.spin_mode == SpinMode.NCL_SOC:
            # NCL: 4 sections (V_total, mag_x, mag_y, mag_z).
            # Only V_total matters for the tunneling barrier.
            # Spin physics enters through the DOS, not the barrier.
            self.data = vol_sections[0]

        else:
            # ISPIN=1: single section
            self.data = vol_sections[0]

        np.save(self.cache_path, self.data)
        print(f"[*] Saved LOCPOT cache to {self.cache_path} with shape {self.data.shape}")
        return self.data
