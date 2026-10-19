# -*- coding: utf-8 -*-
"""
kramers.py — shared Kramers-pairing / HOMO-relative-offset utility.

This module holds the ONE pairing walk used everywhere: the SOC color pairing
in vacupot_plotter.RectAEPAWColorPlotter._build_colors_with_soc_pairing calls
pair_entities() below, and every weight-combining consumer (spread_plot,
spread_collapse, and the mo_ldos_projection package) builds its entities
through the same call — so colors and combining can never disagree.

The walk is the verbatim transplant of the original color-pairing walk:
sort the (band_index_1based, E) pairs by energy; pair adjacent states with
ΔE < thresh; a near-degenerate triplet resolves as pair + single (the walk is
greedy over adjacent sorted states); is_soc False ⇒ every entity is a single.
The rep of a pair is the member the color walk would pick given the same
center_idx argument (the closer-to-pivot member rule), so both call sites
share one rule.

When no INCAR is available to ask (outputs copied without the VASP inputs),
infer_soc_from_spectra below answers the is_soc question from the spectra
themselves: a time-reversal-symmetric NCL/SOC run Kramers-doubles every
spectrum, so the same walk pairs essentially every state. It reuses
pair_entities, so inference and pairing can never disagree either.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# One combined Kramers entity: indices is a tuple of 1 or 2 1-based band
# indices, E_mean is the mean member energy, and rep is the representative
# band index (the member whose color the entity adopts).
Entity = namedtuple("Entity", ["indices", "E_mean", "rep"])


def pair_entities(
    pairs: Sequence[Tuple[int, float]],
    is_soc: bool,
    thresh: float,
    center_idx: Optional[int] = None,
) -> List[Entity]:
    """
    Build the energy-ordered entity list for one component.

    Parameters
    ----------
    pairs : sequence of (band_index_1based, E)
        The component's states (any order; sorted by energy here, exactly as
        the color walk sorts them).
    is_soc : bool
        Gate from detect_ncl_from_incar (or, when no INCAR exists,
        infer_soc_from_spectra). False ⇒ every entity is a single;
        behavior is otherwise unchanged from the unpaired case.
    thresh : float
        Pairing threshold in eV (PlotConfig.soc_pair_thresh_eV): adjacent
        sorted states with ΔE < thresh form one Kramers-pair entity.
    center_idx : int, optional
        The same center/pivot band index the color code passes (the HOMO from
        known_homos, or the get_center_idx fallback). It sets the pivot energy
        for the closer-to-pivot rep rule; when None the middle state's energy
        is the pivot, exactly as in the original color walk.

    Returns
    -------
    list of Entity, ordered by energy (ascending). Each Entity's indices are
    in ascending-energy order; E_mean is the mean of the member energies; rep
    is the member the color walk assigns the pair's color slot to.
    """
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda t: t[1])
    n = len(ordered)

    if not is_soc:
        return [Entity((idx,), float(E), idx) for idx, E in ordered]

    # Resolve pivot energy: center_idx's energy in ordered, else middle state
    # (verbatim rule from _build_colors_with_soc_pairing).
    pivot_E: Optional[float] = None
    if center_idx is not None:
        for idx, E in ordered:
            if idx == center_idx:
                pivot_E = E
                break
    if pivot_E is None:
        pivot_E = ordered[n // 2][1]

    # Walk the sorted list; pair adjacent states with ΔE < thresh. The rep
    # selection below is the verbatim closer-to-pivot member rule from the
    # color walk, so each pair adopts the same member's slot the colors use.
    entities: List[Entity] = []
    i = 0
    while i < n:
        idx_i, E_i = ordered[i]
        if i + 1 < n and (ordered[i + 1][1] - E_i) < thresh:
            idx_j, E_j = ordered[i + 1]
            if center_idx is not None and center_idx in (idx_i, idx_j):
                rep = center_idx
            elif E_j <= pivot_E:
                rep = idx_j  # both below pivot; upper member is closer
            elif E_i >= pivot_E:
                rep = idx_i  # both above pivot; lower member is closer
            else:
                # pair straddles pivot without containing center_idx
                rep = idx_i if abs(E_i - pivot_E) <= abs(E_j - pivot_E) else idx_j
            entities.append(
                Entity((idx_i, idx_j), 0.5 * (float(E_i) + float(E_j)), rep)
            )
            i += 2
        else:
            entities.append(Entity((idx_i,), float(E_i), idx_i))
            i += 1
    return entities


def paired_fraction(
    energies: Sequence[float],
    thresh: float,
) -> float:
    """
    Fraction of the possible Kramers pairs the pairing walk actually forms.

    Runs pair_entities (the ONE walk) with is_soc=True over the energies and
    returns n_pairs_formed / floor(n/2), so a fully Kramers-doubled spectrum
    scores ~1.0 regardless of odd/even state count, and a spectrum with no
    doubling scores ~0.0. Fewer than 2 states scores 0.0.
    """
    n = len(energies)
    if n < 2:
        return 0.0
    ents = pair_entities(
        [(i + 1, float(E)) for i, E in enumerate(energies)], True, thresh
    )
    n_pairs = sum(1 for ent in ents if len(ent.indices) == 2)
    return n_pairs / (n // 2)


def infer_soc_from_spectra(
    spectra: Dict[str, Sequence[float]],
    thresh: float,
    min_states: int = 4,
    min_paired_frac: float = 0.9,
) -> bool:
    """
    Infer SOC (NCL) from Kramers doubling when there is no INCAR to ask.

    A time-reversal-symmetric NCL/SOC run doubles every spectrum: sorted
    adjacent states pair within the SOC pair threshold, so each spectrum's
    paired_fraction is ~1.0 (Kramers partners are exactly degenerate at the
    Gamma point; the txt files round to 1e-4 eV, far below any sane thresh).
    Without SOC only accidental/symmetry degeneracies pair, and the fraction
    stays far below 1.

    Parameters
    ----------
    spectra : {name: energies}
        One energy list per spectrum to test — the per-component state
        energies (comp_pairs) plus optionally the full-system energies.
    thresh : float
        The SOC pairing threshold (PlotConfig.soc_pair_thresh_eV), so the
        inference keys on exactly the degeneracy the pairing will use.
    min_states : int
        Spectra with fewer states than this carry too little evidence and are
        ignored.
    min_paired_frac : float
        Decision level for the MEDIAN paired_fraction over the qualifying
        spectra. The median makes one noisy spectrum (e.g. a dense metal
        manifold with a few numerically split pairs) unable to flip the
        verdict in either direction.

    Returns
    -------
    bool — True when the median paired_fraction of the qualifying spectra
    reaches min_paired_frac; False when it does not or no spectrum qualifies.
    """
    fracs = [
        paired_fraction(E, thresh)
        for E in spectra.values()
        if len(E) >= min_states
    ]
    if not fracs:
        return False
    return float(np.median(fracs)) >= min_paired_frac


def homo_entity_offsets(
    entities: Sequence[Entity],
    homo_index: Optional[int],
    fermi: Optional[float] = None,
) -> Dict[int, Entity]:
    """
    Map HOMO-relative offsets onto the energy-ordered entity list.

    Offset 0 is the entity whose indices contain homo_index (from
    meta/homo_indices / the '# HOMOS:' header); negative offsets walk down
    the energy-ordered entity list, positive up (+1 = LUMO entity).

    If homo_index is absent (None, or contained in no entity), fall back to
    the highest entity with E_mean <= fermi — matching get_center_idx logic,
    which selects the highest occupied state on the same energy scale (its
    threshold is 0.0 on the aligned axis; pass the component Fermi via
    `fermi` to use it instead). If that also fails, raise.
    """
    entities = list(entities)
    h: Optional[int] = None
    if homo_index is not None:
        target = int(homo_index)
        for k, ent in enumerate(entities):
            if target in ent.indices:
                h = k
                break
    if h is None:
        thresh = 0.0 if fermi is None else float(fermi)
        occupied = [k for k, ent in enumerate(entities) if ent.E_mean <= thresh]
        if occupied:
            h = max(occupied)  # entities are energy-ordered: highest E_mean
    if h is None:
        raise ValueError(
            "homo_entity_offsets: homo_index %r is in no entity and no entity "
            "has E_mean <= %s — cannot anchor offset 0." % (homo_index, fermi)
        )
    return {k - h: ent for k, ent in enumerate(entities)}


def combine_pair_weights(
    grid_label: np.ndarray,
    entities: Sequence[Entity],
) -> np.ndarray:
    """
    Combine per-state weights into per-entity weights over full states:
        p_ent(j) = sum_{i in entity.indices} p_i(j)

    Parameters
    ----------
    grid_label : ndarray, shape (n_comp, n_full)
        grid/{label} rows (or txt-derived per-state weights assembled into
        the same layout): row i-1 holds band i's weights over full states.
    entities : sequence of Entity
        The component's entity list from pair_entities.

    Returns
    -------
    ndarray, shape (len(entities), n_full): row k is entity k's combined
    weight over full states, in the entities' (energy-ordered) row order.
    """
    G = np.asarray(grid_label)
    out = np.zeros((len(entities), G.shape[1]), dtype=G.dtype)
    for k, ent in enumerate(entities):
        for idx in ent.indices:
            out[k, :] += G[int(idx) - 1, :]
    return out
