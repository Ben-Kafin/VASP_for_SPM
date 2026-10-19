#!/usr/bin/env python3

import os
from PROJCAR_parser import ProjcarParser

class LocprojAggregator:
    """
    Aggregate non-zero projected orbitals per atom (site) from PROJCAR,
    then group atoms by identical orbital sets and emit LOCPROJ lines.
    """

    def __init__(self, directory):
        self.directory   = directory
        self.parser      = ProjcarParser(directory)
        self.final_data  = self.parser.parse()
        self.site_orb_map = self._collect_nonzero_by_site()
        self.groups       = self._group_sites_by_orbitals()

    def _collect_nonzero_by_site(self):
        site_map = {}
        for bands in self.final_data.values():
            for site_blocks in bands.values():
                for site, spins in site_blocks.items():
                    for block in spins.values():
                        for orb, (v1, v2) in block["orbitals"].items():
                            if v1 and v2 and (abs(v1) > 0 or abs(v2) > 0):
                                site_map.setdefault(site, set()).add(orb)
        return {s: sorted(orbs) for s, orbs in site_map.items()}

    def _group_sites_by_orbitals(self):
        grouping = {}
        for site, orbs in self.site_orb_map.items():
            grouping.setdefault(tuple(orbs), []).append(site)
        return {orbs: sorted(sites) for orbs, sites in grouping.items()}

    def print_summary(self):
        print("\nUnique non-zero orbitals per site:")
        for site, orbs in sorted(self.site_orb_map.items()):
            print(f"  Site {site}: {', '.join(orbs)}")

        print("\nLOCPROJ lines (grouped by identical sets):")
        for orbs, sites in self.groups.items():
            print(f"  LOCPROJ = {' '.join(map(str, sites))} : {' '.join(orbs)}")



# -----------------------------
# Run section — edit this path:
projcar_dir = r'dir/run/'

agg = LocprojAggregator(projcar_dir)
agg.print_summary()