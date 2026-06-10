"""truth_matching.py - gen-level truth matching of reconstructed tri-jets.

For all-hadronic t-tbar (and more generally a pair of "tri-jet" resonances) each
parent decays to three quarks: here ``b`` plus the two quarks from ``W -> q q'``.
:func:`truth_matched_trijets` matches the reconstructed ``ScoutingPFJet`` jets to
those gen quarks and, when a parent's three quarks are each matched to a distinct
reco jet, returns the summed reco four-vector of that parent (the "tri-jet").
Per event you therefore get **0, 1 or 2** four-vectors -- one per parent that is
fully reconstructed; an event with no usable truth interpretation yields nothing.

The matching mirrors the examples in
``run3-scouting-multijets/MultijetsML/nano_to_h5.py`` (and ``passwd_abc/``), with
one adaptation. ScoutingNano stores ``GenPart_status`` (the Pythia/HepMC status
code) but **not** the NanoAOD ``statusFlags`` bitfield, so we cannot call
``hasFlags(["isLastCopy"])`` / ``distinctChildren``. Instead we select the
hard-process outgoing partons via ``GenPart_status == 23`` and walk the decay
tree with ``GenPart_genPartIdxMother``:

  * ``b``  from top:  ``status==23``, ``|pdgId|==5``, mother is a top (``|pdgId|==6``)
  * ``q q'`` from W:  ``status==23``, a quark,        mother is a W   (``|pdgId|==24``)

The sign of the ``b`` / ``W`` splits the six quarks into the two parents
(top vs. antitop). These are the *first* (pre-shower) hard copies; for a
``dR < 0.4`` jet match their direction is equivalent to the last copy used by the
NanoAOD-flag version.

Works on an *eager* awkward ``events`` array -- e.g. a coffea
``NanoEventsFactory`` with ``BaseSchema`` (flat ``GenPart_*`` / ``ScoutingPFJet_*``
branches) or a nested NanoAOD record. No dask is required.
"""

import numpy as np
import awkward as ak
import vector

vector.register_awkward()  # enables Momentum4D behaviors (.deltaR, .px, .mass, ...)

# --- Pythia status code of outgoing partons of the hardest subprocess. The b
# from the top and the q q' from the W carry this status in the full gen record.
HARD_STATUS = 23
TOP_PDGID = 6
W_PDGID = 24
B_PDGID = 5

__all__ = ["truth_matched_trijets"]


def _get(events, name):
    """Return ``events.<name>`` across flat / nested / dotted layouts, else None.

    Handles both the slimmer/evaluator flat branch names (``GenPart_pt``) and a
    nested NanoAOD record (``events.GenPart.pt``). Returns ``None`` when absent so
    callers can degrade gracefully (e.g. data with no GenPart).
    """
    if name in events.fields:
        return events[name]
    coll, _, sub = name.partition("_")
    if coll and sub and coll in events.fields:
        rec = events[coll]
        subfields = getattr(rec, "fields", None)
        if subfields and sub in subfields:
            return rec[sub]
    return None


def _momentum4d(pt, eta, phi, mass):
    """Build a Momentum4D awkward array from (pt, eta, phi, mass) collections."""
    return ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": mass},
        with_name="Momentum4D",
    )


def _nearest_dr(jets, partons):
    """Per jet, the ΔR to its nearest parton (None where there are no partons).

    Shapes: jets ``(event, njet)``, partons ``(event, nparton)`` ->
    result ``(event, njet)``.
    """
    pair = ak.cartesian({"j": jets, "p": partons}, nested=True)  # (ev, njet, nparton)
    dr = pair["j"].deltaR(pair["p"])
    return ak.min(dr, axis=-1)  # nearest parton per jet; None when nparton == 0


def _empty_result(n_events):
    """A length-``n_events`` jagged Momentum4D array with 0 entries per event."""
    zeros = np.zeros(n_events, dtype=np.float64)
    none_vec = ak.mask(
        ak.zip(
            {"px": zeros, "py": zeros, "pz": zeros, "energy": zeros},
            with_name="Momentum4D",
        ),
        np.zeros(n_events, dtype=bool),  # mask everything -> all None
    )
    return ak.concatenate([ak.singletons(none_vec), ak.singletons(none_vec)], axis=1)


def truth_matched_trijets(
    events,
    *,
    dr_max=0.4,
    jet_pt_min=30.0,
    jet_abseta_max=2.4,
    hard_status=HARD_STATUS,
    jet_pt_branch="ScoutingPFJet_pt",
    jet_m_branch="ScoutingPFJet_m",
):
    """Truth-match reconstructed tri-jets in a coffea ``events`` array.

    Parameters
    ----------
    events : awkward.Array
        Eager coffea NanoEvents-style array exposing ``ScoutingPFJet_*`` and
        ``GenPart_*`` (flat or nested). Must carry ``GenPart_status`` and
        ``GenPart_genPartIdxMother`` for the gen-tree walk.
    dr_max : float, default 0.4
        Maximum ΔR for a reco jet to count as matched to a gen quark.
    jet_pt_min : float or None, default 30.0
        Minimum jet pT (GeV) before matching; ``None`` disables the cut.
    jet_abseta_max : float or None, default 2.4
        Maximum jet ``|eta|`` before matching; ``None`` disables the cut.
    hard_status : int, default 23
        Pythia status of the hard outgoing partons to match against. The single
        knob to retune if a sample stores the decay quarks under another status.
    jet_pt_branch, jet_m_branch : str
        Branches supplying the jet pT / mass, e.g. ``"ScoutingPFJet_pt_raw"`` /
        ``"ScoutingPFJet_m_raw"`` for pre-JEC kinematics, or the ``_jesUp`` /
        ``_jerDown`` variants for systematics. Direction (eta/phi) always comes
        from ``ScoutingPFJet_eta`` / ``ScoutingPFJet_phi``.

    Returns
    -------
    awkward.Array
        Length ``len(events)``; each event holds 0-2 ``Momentum4D`` four-vectors,
        one per parent (top / antitop) whose three quarks were each matched to a
        distinct reco jet. The four-vector is the **sum of those three jets** (the
        reconstructed parent). Events without a usable truth interpretation -- no
        GenPart, not enough matched jets -- contribute an empty list.
    """
    n_events = len(events)

    jet_pt = _get(events, jet_pt_branch)
    jet_eta = _get(events, "ScoutingPFJet_eta")
    jet_phi = _get(events, "ScoutingPFJet_phi")
    jet_m = _get(events, jet_m_branch)

    gen_pt = _get(events, "GenPart_pt")
    gen_eta = _get(events, "GenPart_eta")
    gen_phi = _get(events, "GenPart_phi")
    gen_mass = _get(events, "GenPart_mass")
    gen_pdg = _get(events, "GenPart_pdgId")
    gen_status = _get(events, "GenPart_status")
    gen_mother = _get(events, "GenPart_genPartIdxMother")

    # No reco jets or no gen-truth handles -> nothing can be matched.
    if any(x is None for x in (jet_pt, jet_eta, jet_phi, jet_m)):
        return _empty_result(n_events)
    if any(
        x is None
        for x in (gen_pt, gen_eta, gen_phi, gen_mass, gen_pdg, gen_status, gen_mother)
    ):
        return _empty_result(n_events)

    # Normalize fixed-width (regular) layouts to jagged: on a regular array a
    # boolean mask flattens numpy-style, destroying the per-event structure.
    jet_pt, jet_eta, jet_phi, jet_m = (
        ak.from_regular(x, axis=-1) for x in (jet_pt, jet_eta, jet_phi, jet_m)
    )
    gen_pt, gen_eta, gen_phi, gen_mass, gen_pdg, gen_status, gen_mother = (
        ak.from_regular(x, axis=-1)
        for x in (gen_pt, gen_eta, gen_phi, gen_mass, gen_pdg, gen_status, gen_mother)
    )

    # --- reco jets (preselected) as four-vectors ---------------------------
    jet_mask = ak.ones_like(jet_pt, dtype=bool)
    if jet_pt_min is not None:
        jet_mask = jet_mask & (jet_pt > jet_pt_min)
    if jet_abseta_max is not None:
        jet_mask = jet_mask & (abs(jet_eta) < jet_abseta_max)
    jets = _momentum4d(
        jet_pt[jet_mask], jet_eta[jet_mask], jet_phi[jet_mask], jet_m[jet_mask]
    )

    # --- gen quarks of the two parents, split by b / W charge --------------
    # pdgId of each particle's mother (0 where there is no mother, idx == -1).
    safe_mother = ak.where(gen_mother < 0, 0, gen_mother)
    mother_pdg = ak.where(gen_mother < 0, 0, gen_pdg[safe_mother])

    is_hard = gen_status == hard_status
    absid = abs(gen_pdg)
    is_quark = (absid >= 1) & (absid <= 5)

    # parent 1 = top:     b(+5)  <- top(+6),  q q' <- W+(+24)
    # parent 2 = antitop: b(-5)  <- top(-6),  q q' <- W-(-24)
    group1 = is_hard & (
        ((gen_pdg == B_PDGID) & (mother_pdg == TOP_PDGID))
        | (is_quark & (mother_pdg == W_PDGID))
    )
    group2 = is_hard & (
        ((gen_pdg == -B_PDGID) & (mother_pdg == -TOP_PDGID))
        | (is_quark & (mother_pdg == -W_PDGID))
    )

    partons1 = _momentum4d(
        gen_pt[group1], gen_eta[group1], gen_phi[group1], gen_mass[group1]
    )
    partons2 = _momentum4d(
        gen_pt[group2], gen_eta[group2], gen_phi[group2], gen_mass[group2]
    )

    # --- assign each jet to the closer parent (within dr_max) --------------
    d1 = ak.fill_none(_nearest_dr(jets, partons1), np.inf)
    d2 = ak.fill_none(_nearest_dr(jets, partons2), np.inf)
    sel1 = (d1 < dr_max) & (d1 <= d2)  # closer to a parent-1 quark (ties -> 1)
    sel2 = (d2 < dr_max) & (d2 < d1)  # strictly closer to a parent-2 quark

    # --- a parent is reconstructed iff exactly 3 jets are matched to it ----
    def _trijet(sel):
        n = ak.sum(sel, axis=1)
        vec = ak.zip(
            {
                "px": ak.sum(jets.px[sel], axis=1),
                "py": ak.sum(jets.py[sel], axis=1),
                "pz": ak.sum(jets.pz[sel], axis=1),
                "energy": ak.sum(jets.energy[sel], axis=1),
            },
            with_name="Momentum4D",
        )
        return ak.mask(vec, n == 3)  # None unless exactly three distinct jets

    t1 = _trijet(sel1)
    t2 = _trijet(sel2)

    # singletons drops the None parents -> 0-2 four-vectors per event.
    return ak.concatenate([ak.singletons(t1), ak.singletons(t2)], axis=1)
