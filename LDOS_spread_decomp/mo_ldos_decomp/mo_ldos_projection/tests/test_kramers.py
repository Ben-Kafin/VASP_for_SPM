# -*- coding: utf-8 -*-
"""Section 10.5 — Kramers/HOMO utilities: pairing identical to the color
walk (refactor regression), offset 0 entity contains homo_indices[label],
triplet resolves pair + single."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from mo_ldos_projection._imports import (
    pair_entities, homo_entity_offsets, combine_pair_weights)


def _reference_color_walk(pairs, center_idx, thresh):
    """Verbatim re-statement of the ORIGINAL inline color-pairing walk
    (pre-refactor _build_colors_with_soc_pairing) producing idx→rep."""
    ordered = sorted(pairs, key=lambda t: t[1])
    n = len(ordered)
    pivot_E: Optional[float] = None
    if center_idx is not None:
        for idx, E in ordered:
            if idx == center_idx:
                pivot_E = E
                break
    if pivot_E is None:
        pivot_E = ordered[n // 2][1]
    idx_to_rep: Dict[int, int] = {}
    i = 0
    while i < n:
        idx_i, E_i = ordered[i]
        if i + 1 < n and (ordered[i + 1][1] - E_i) < thresh:
            idx_j, E_j = ordered[i + 1]
            if center_idx is not None and center_idx in (idx_i, idx_j):
                rep = center_idx
            elif E_j <= pivot_E:
                rep = idx_j
            elif E_i >= pivot_E:
                rep = idx_i
            else:
                rep = idx_i if abs(E_i - pivot_E) <= abs(E_j - pivot_E) else idx_j
            idx_to_rep[idx_i] = rep
            idx_to_rep[idx_j] = rep
            i += 2
        else:
            idx_to_rep[idx_i] = idx_i
            i += 1
    return idx_to_rep


def test_pairing_identical_to_color_walk_on_example(prepared):
    P, cfg, plotter = prepared
    for label, pairs in P.comp_pairs.items():
        center = P.known_homos.get(label)
        ref = _reference_color_walk(pairs, center, cfg.soc_pair_thresh_eV)
        ents = pair_entities(pairs, True, cfg.soc_pair_thresh_eV,
                             center_idx=center)
        got = {}
        for ent in ents:
            for idx in ent.indices:
                got[idx] = ent.rep
        assert got == ref, label


def test_colors_route_through_shared_walk(prepared):
    """The color maps produced after the refactor equal the ones the
    original inline walk would produce (per-index equality)."""
    P, cfg, plotter = prepared
    for label in P.comp_pairs:
        pairs = P.comp_pairs[label]
        center = P.known_homos.get(label)
        is_metal = label.lower() == "metal"
        cmap_name = cfg.cmap_name_metal if is_metal else cfg.cmap_name_simple
        p_neg = cfg.power_metal_neg if is_metal else cfg.power_simple_neg
        p_pos = cfg.power_metal_pos if is_metal else cfg.power_simple_pos
        got = plotter._build_colors_with_soc_pairing(
            pairs, cmap_name, center, p_neg, p_pos, True,
            cfg.soc_pair_thresh_eV)
        base = plotter._build_colors_rank_pivot(
            pairs, cmap_name, center, p_neg, p_pos)
        ref_map = _reference_color_walk(pairs, center, cfg.soc_pair_thresh_eV)
        expect = {idx: base[rep] for idx, rep in ref_map.items()}
        assert got == expect, label


def test_offset_zero_contains_homo(prepared):
    P, cfg, _ = prepared
    for label, pairs in P.comp_pairs.items():
        homo = P.known_homos.get(label)
        if homo is None:
            continue
        ents = pair_entities(pairs, P.is_soc, cfg.soc_pair_thresh_eV,
                             center_idx=homo)
        offs = homo_entity_offsets(ents, homo)
        assert homo in offs[0].indices, label


def test_triplet_resolves_pair_plus_single():
    ents = pair_entities([(1, 0.0), (2, 0.0004), (3, 0.0008)], True, 0.001)
    assert [e.indices for e in ents] == [(1, 2), (3,)]


def test_non_soc_all_singles():
    pairs = [(1, 0.0), (2, 0.0001), (3, 5.0)]
    ents = pair_entities(pairs, False, 0.001)
    assert [e.indices for e in ents] == [(1,), (2,), (3,)]
    assert [e.rep for e in ents] == [1, 2, 3]


def test_homo_offsets_fallback_and_raise():
    ents = pair_entities([(1, -2.0), (2, -1.0), (3, 1.0)], False, 0.001)
    # fallback: highest entity with E_mean <= 0 anchors offset 0
    offs = homo_entity_offsets(ents, None)
    assert offs[0].indices == (2,)
    with pytest.raises(ValueError):
        homo_entity_offsets(
            pair_entities([(1, 1.0)], False, 0.001), None)


def test_combine_pair_weights_sums_rows():
    G = np.arange(12, dtype=float).reshape(4, 3)
    ents = pair_entities([(1, 0.0), (2, 0.0001), (3, 1.0), (4, 2.0)],
                         True, 0.001)
    W = combine_pair_weights(G, ents)
    assert W.shape == (3, 3)
    np.testing.assert_array_equal(W[0], G[0] + G[1])
    np.testing.assert_array_equal(W[1], G[2])
    np.testing.assert_array_equal(W[2], G[3])
