"""Joint compound-assignment + K estimation (multi-task, two losses).

Compound labels are sparse, but compounds separate strongly in heat-per-G².
Rather than classify-then-refit, this fits a single joint objective per
car and axle:

- latent variable: the axle's compound for each session;
- supervised loss: labeled/seeded sessions are pinned to their compound
  (responsibility 1);
- regression loss: every lap contributes
  ``(ΔT − K[compound, corner, cond] · x)²`` with
  ``x = g²·c_track·warmup_frac``, weighted by the session's compound
  responsibilities.

Under Gaussian residuals the optimum is a mixture of linear regressions,
solved by EM in closed form: the E-step computes per-(session, axle)
posteriors over compounds from the lap residuals; the M-step refits each
``K[car, compound, corner, condition]`` by responsibility-weighted least
squares through the origin. Iterated to convergence.

Ground rules:
- Human/seeded labels always win (pinned); inference never overrides.
- Soft assignments are TRAINING-only. Held-out evaluation uses human/seed
  labels exclusively — classifying a held-out session from its own
  temperatures would leak the target into the model choice.
- A car participates only when at least two compounds each have at least
  ``MIN_LABELED_SESSIONS`` labeled sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_LABELED_SESSIONS = 2  # per (car, compound) for that car to participate
MIN_LAPS_PER_AXLE = 5  # session laps needed for a meaningful posterior
HARD_ASSIGN_RESP = 0.9  # report threshold for the audit CLI
SIGMA2_FLOOR = 4.0  # °C² — residual-variance floor so the E-step stays sane
# Laps within a session share setup, driver and weather, so they are far
# from independent draws: temper each session's posterior to this many
# effective observations. Without it the E-step saturates (softmax over a
# sum of dozens of laps) and even genuinely ambiguous sessions get claimed
# with certainty, dragging a cluster toward them.
SESSION_EFF_SAMPLES = 4.0
# Outlier gate: a free session whose BEST cluster still fits badly (mean
# per-lap chi² above this) matches no known compound — e.g. a tire we have
# no labels for — and must stay unlabeled rather than be absorbed into the
# nearest cluster and drag its K.
OUTLIER_CHI2_PER_LAP = 9.0
# A (compound, corner, condition) bucket only exists when pinned sessions
# put at least this many laps in it. Free sessions refine anchored buckets
# but cannot spawn new ones — otherwise a symmetric-fixed-point forms where
# unlabeled laps create identical K for every compound in a bucket no
# pinned data covers, and the E-step can never tell them apart.
MIN_PINNED_LAPS_PER_BUCKET = 5.0
EM_MAX_ITER = 30
EM_TOL = 1e-4

# Every run has ONE tire on all four corners (team rule): the latent unit
# is the whole session, and all four corners' laps vote on one posterior.
_AXLE_CORNERS = {"all": ("fl", "fr", "rl", "rr")}
_CORNER_AXLE = {c: a for a, cs in _AXLE_CORNERS.items() for c in cs}


@dataclass(frozen=True)
class AxleAssignment:
    session_id: str
    car: str
    axle: str
    compound: str  # argmax compound
    responsibility: float
    n_laps: int
    pinned: bool  # True when from a human/seed label
    poor_fit: bool = False  # best compound still fits badly (unknown tire?)


def _suff_stats(
    laps_for_fit: pd.DataFrame,
    tau_by_car_corner_cond: dict,
    c_track_by_track: dict,
) -> pd.DataFrame:
    """Per-(session, corner, condition) sufficient statistics.

    Columns: session_id, car, corner, condition, sxx, sxy, syy, n where
    x = g²·c_track·warmup_frac and y = ΔT.
    """
    laps = laps_for_fit.copy()
    laps["g2_lap"] = laps["heat_proxy"] / laps["on_track_s"]
    laps = laps[
        laps["g2_lap"].notna()
        & (laps["g2_lap"] > 0)
        & (laps["t_cum_s"] > 0)
        & (laps["condition"] != "unknown")
    ]

    def c_val(track: object) -> float:
        fp = c_track_by_track.get(str(track))
        return fp.value if fp is not None else 1.0

    rows = []
    for corner in _CORNER_AXLE:
        delta_col = f"delta_t_{corner}"
        sub = laps[laps[delta_col].notna()]
        for (sid, car, cond), grp in sub.groupby(["session_id", "car", "condition"]):
            tau_fp = tau_by_car_corner_cond.get(
                (str(car), corner, str(cond))
            ) or tau_by_car_corner_cond.get((str(car), corner, "dry"))
            if tau_fp is None or tau_fp.value <= 0:
                continue
            frac = 1.0 - np.exp(-grp["t_cum_s"].to_numpy() / tau_fp.value)
            x = grp["g2_lap"].to_numpy() * grp["track_canonical"].map(c_val).to_numpy() * frac
            y = grp[delta_col].to_numpy(dtype=float)
            ok = x > 0
            if not ok.any():
                continue
            x, y = x[ok], y[ok]
            rows.append(
                {
                    "session_id": sid,
                    "car": str(car),
                    "corner": corner,
                    "condition": str(cond),
                    "sxx": float(np.sum(x * x)),
                    "sxy": float(np.sum(x * y)),
                    "syy": float(np.sum(y * y)),
                    "n": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def fit_compounds_em(
    laps_for_fit: pd.DataFrame,
    labels: pd.DataFrame,
    tau_by_car_corner_cond: dict,
    c_track_by_track: dict,
    *,
    max_iter: int = EM_MAX_ITER,
) -> tuple[
    dict[tuple[str, str, str, str], tuple[float, float, float]],
    list[AxleAssignment],
    dict[str, dict[str, float]],
]:
    """Joint EM over compound assignments and a DECOMPOSED K.

    K is structured, not free per bucket:

        K_effective = c_track[track] · K_base[car, corner, cond] · m[car, compound]

    (c_track is already inside the regressor ``x``, so this function fits
    ``K_base`` per (car, corner, condition) and one scalar multiplier per
    (car, compound), by responsibility-weighted alternating least squares
    inside each M-step.) The structure pools statistical strength — every
    lap of every compound informs K_base, and each compound's m is shared
    across corners and conditions — and it removes the symmetric fixed
    point free-bucket fitting suffered from. Identifiability: the
    lap-weighted geometric mean of m over a car's compounds is 1, so
    K_base is the car's "average tire" and m the compound's ratio.

    Selection is FORCED: every eligible (session, axle) carries a posterior
    over the car's compounds (pinned one-hot where labeled/seeded, softmax
    elsewhere) — there is no unlabeled pool inside a participating car.
    Sessions whose best fit is still poor are flagged ``poor_fit`` (audit:
    possible unknown tire) but still assigned softly.

    Returns ``(k_by_compound, assignments, multipliers)`` where
    ``k_by_compound`` maps ``(car, compound, corner, condition) ->
    (K, stderr, n_effective)`` (the base·m product, ready for the artifact),
    and ``multipliers`` maps ``car -> {compound: m}``.
    """
    if laps_for_fit.empty or labels.empty:
        return {}, [], {}

    stats = _suff_stats(laps_for_fit, tau_by_car_corner_cond, c_track_by_track)
    if stats.empty:
        return {}, [], {}
    stats["axle"] = stats["corner"].map(_CORNER_AXLE)

    pinned: dict[tuple[str, str], str] = {}
    for r in labels.itertuples():
        if isinstance(r.compound, str) and r.compound:
            pinned[(str(r.session_id), "all")] = r.compound

    out_k: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    assignments: list[AxleAssignment] = []
    multipliers: dict[str, dict[str, float]] = {}

    for car_key, car_stats in stats.groupby("car"):
        car = str(car_key)
        pinned_here: dict[str, set] = {}
        for (sid, axle), comp in pinned.items():
            if sid in set(car_stats["session_id"]):
                pinned_here.setdefault(comp, set()).add(sid)
        compounds = sorted(
            c for c, sids in pinned_here.items() if len(sids) >= MIN_LABELED_SESSIONS
        )
        if len(compounds) < 2:
            compounds = sorted(pinned_here)
            if not compounds:
                continue

        unit_lap_counts: dict[tuple[str, str], int] = {
            (str(k[0]), str(k[1])): int(v)  # type: ignore[index]
            for k, v in car_stats.groupby(["session_id", "axle"])["n"].sum().items()
        }
        units = [u for u, n in unit_lap_counts.items() if n >= MIN_LAPS_PER_AXLE or u in pinned]
        if not units:
            continue
        unit_idx = {u: i for i, u in enumerate(units)}
        n_units, n_comp = len(units), len(compounds)
        comp_idx = {c: j for j, c in enumerate(compounds)}

        # Forced selection: every unit starts uniform (pinned -> one-hot).
        resp = np.full((n_units, n_comp), 1.0 / n_comp)
        pin_mask = np.zeros(n_units, dtype=bool)
        for u, i in unit_idx.items():
            pcomp = pinned.get(u)
            if pcomp is not None and pcomp in comp_idx:
                resp[i] = 0.0
                resp[i, comp_idx[pcomp]] = 1.0
                pin_mask[i] = True

        s_unit = car_stats.copy()
        s_unit["unit"] = list(zip(s_unit["session_id"], s_unit["axle"]))
        s_unit = s_unit[s_unit["unit"].isin(unit_idx)]
        s_unit["ui"] = s_unit["unit"].map(unit_idx)
        buckets = list(s_unit.groupby(["corner", "condition"]))
        bucket_arrays = []
        for bkey, grp in buckets:
            corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
            bucket_arrays.append(
                (
                    corner,
                    cond,
                    grp["ui"].to_numpy(),
                    grp["sxx"].to_numpy(),
                    grp["sxy"].to_numpy(),
                    grp["syy"].to_numpy(),
                    grp["n"].to_numpy(),
                )
            )

        sigma2: dict[str, float] = {c: 25.0 for c in _CORNER_AXLE}
        base: dict[tuple[str, str], float] = {}
        m = np.ones(n_comp)
        # Robust weights: selection is forced, but a free unit's INFLUENCE
        # on (base, m) shrinks as its best-compound fit degrades — a
        # mid-cluster or unknown-tire session is still assigned (and
        # flagged poor_fit) without dragging any cluster toward itself.
        robust_w = np.ones(n_units)
        n_u_arr = np.array([max(float(unit_lap_counts.get(u, 1)), 1.0) for u in units])
        temper = np.minimum(1.0, SESSION_EFF_SAMPLES / n_u_arr)[:, None]
        free = ~pin_mask

        prev_ll = -np.inf
        for _ in range(max_iter):
            # ---- M-step: alternating LS on (base, m) ----
            resp_eff = resp * robust_w[:, None]
            for _als in range(3):
                for corner, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
                    num = den = 0.0
                    for j in range(n_comp):
                        w = resp_eff[ui, j]
                        num += m[j] * float(np.sum(w * sxy))
                        den += m[j] * m[j] * float(np.sum(w * sxx))
                    if den > 0:
                        base[(corner, cond)] = num / den
                for j in range(n_comp):
                    num = den = 0.0
                    for corner, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
                        b = base.get((corner, cond))
                        if b is None:
                            continue
                        w = resp_eff[ui, j]
                        num += b * float(np.sum(w * sxy))
                        den += b * b * float(np.sum(w * sxx))
                    if den > 0:
                        m[j] = num / den
                # Anchor: lap-weighted geometric mean of m == 1.
                w_c = np.array(
                    [max(float(np.sum(resp[:, j] * n_u_arr)), 1e-9) for j in range(n_comp)]
                )
                m = np.clip(m, 1e-3, None)
                g = float(np.exp(np.sum(w_c * np.log(m)) / np.sum(w_c)))
                if g > 0:
                    m = m / g
                    for key in list(base):
                        base[key] *= g

            def k_for_scoring(j: int, corner: str, cond: str) -> float | None:
                for c2 in (cond, "dry"):
                    b = base.get((corner, c2))
                    if b is not None:
                        return b * float(m[j])
                for (c_corner, _c2), b in base.items():
                    if c_corner == corner:
                        return b * float(m[j])
                return None

            # Residual variance per corner from pinned units only (kept
            # clean of whatever the free units turn out to be).
            for corner in _CORNER_AXLE:
                num = den = 0.0
                for c2, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
                    if c2 != corner:
                        continue
                    for j in range(n_comp):
                        k = k_for_scoring(j, corner, cond)
                        if k is None:
                            continue
                        w = resp[ui, j] * pin_mask[ui]
                        rss_arr = syy - 2 * k * sxy + k * k * sxx
                        num += float(np.sum(w * rss_arr))
                        den += float(np.sum(w * n_arr))
                if den > 0:
                    sigma2[corner] = max(num / den, SIGMA2_FLOOR)

            # ---- E-step ----
            loglik = np.zeros((n_units, n_comp))
            covered = np.zeros((n_units, n_comp))
            for corner, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
                s2 = sigma2[corner]
                for j in range(n_comp):
                    k = k_for_scoring(j, corner, cond)
                    if k is None:
                        continue
                    rss_arr = syy - 2 * k * sxy + k * k * sxx
                    np.add.at(loglik[:, j], ui, -0.5 * rss_arr / s2)
                    np.add.at(covered[:, j], ui, n_arr)
            loglik = np.where(covered > 0, loglik, -np.inf)
            if free.any():
                ll = (loglik * temper)[free]
                row_max = ll.max(axis=1, keepdims=True)
                ok_rows = np.isfinite(row_max[:, 0])
                p = np.full_like(ll, 1.0 / n_comp)
                if ok_rows.any():
                    z = np.exp(ll[ok_rows] - row_max[ok_rows])
                    p[ok_rows] = z / z.sum(axis=1, keepdims=True)
                resp[free] = p
            # Refresh robust weights for free units from best-fit quality.
            with np.errstate(invalid="ignore"):
                best_chi2_iter = np.where(
                    np.isfinite(loglik).any(axis=1),
                    -2.0
                    * np.nanmax(np.where(np.isfinite(loglik), loglik, np.nan), axis=1)
                    / n_u_arr,
                    np.inf,
                )
            new_w = np.minimum(1.0, OUTLIER_CHI2_PER_LAP / np.maximum(best_chi2_iter, 1e-9))
            robust_w = np.where(pin_mask, 1.0, new_w)
            total_ll = float(np.sum(np.where(resp > 0, loglik * resp, 0.0)))
            if abs(total_ll - prev_ll) < EM_TOL * (1 + abs(prev_ll)):
                break
            prev_ll = total_ll

        # Poor-fit flag (possible unknown tire): best cluster still fits
        # badly against pinned-only variance. Selection stays forced — the
        # flag is for the audit, not an exclusion.
        chi2_num = np.zeros((n_units, n_comp))
        for corner, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
            for j in range(n_comp):
                k = k_for_scoring(j, corner, cond)
                if k is None:
                    chi2_num[:, j] += np.inf
                    continue
                rss_arr = syy - 2 * k * sxy + k * k * sxx
                np.add.at(chi2_num[:, j], ui, rss_arr / sigma2[corner])
        best_chi2 = chi2_num.min(axis=1) / n_u_arr
        poor = best_chi2 > OUTLIER_CHI2_PER_LAP

        # ---- Export effective K = base·m with stderr + effective n ----
        for corner, cond, ui, sxx, sxy, syy, n_arr in bucket_arrays:
            b = base.get((corner, cond))
            if b is None:
                continue
            for comp, j in comp_idx.items():
                k = b * float(m[j])
                w = (resp * robust_w[:, None])[ui, j]
                sxx_w = float(np.sum(w * sxx))
                n_eff = float(np.sum(w * n_arr))
                if sxx_w <= 0 or n_eff < MIN_LAPS_PER_AXLE:
                    continue
                rss = float(np.sum(w * (syy - 2 * k * sxy + k * k * sxx)))
                stderr = float(np.sqrt(max(rss, 0.0) / max(n_eff - 1, 1.0) / sxx_w))
                out_k[(car, comp, corner, cond)] = (k, stderr, n_eff)

        multipliers[car] = {comp: round(float(m[j]), 4) for comp, j in comp_idx.items()}
        for u, i in unit_idx.items():
            j = int(np.argmax(resp[i]))
            assignments.append(
                AxleAssignment(
                    session_id=u[0],
                    car=car,
                    axle=u[1],
                    compound=compounds[j],
                    responsibility=round(float(resp[i, j]), 3),
                    n_laps=unit_lap_counts.get(u, 0),
                    pinned=bool(pin_mask[i]),
                    poor_fit=bool(poor[i]),
                )
            )

    return out_k, assignments, multipliers


def apply_condition_seeds(
    labels: pd.DataFrame,
    laps: pd.DataFrame,
    seeds: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Expand declarative condition seeds into per-session labels.

    ``seeds`` (from the sidecar's ``condition_seeds`` block) maps
    ``car -> {condition -> compound}`` — e.g. every all-dry KK-SII session
    seeds as DRY tires, every all-wet one as WET. Sessions with mixed or
    unlisted conditions are left alone, and existing labels always win.
    """
    if not seeds or laps.empty:
        return labels
    have = set(labels["session_id"]) if not labels.empty else set()
    lp = laps[laps["condition"] != "unknown"]
    conds = lp.groupby(["session_id", "car"])["condition"].agg(set).reset_index()
    rows = []
    for sid, car, condset in zip(conds["session_id"], conds["car"], conds["condition"]):
        if sid in have or len(condset) != 1:
            continue
        mapping = seeds.get(str(car))
        if not mapping:
            continue
        compound = mapping.get(next(iter(condset)))
        if compound:
            rows.append({"session_id": sid, "compound": compound, "source": "seed"})
    if not rows:
        return labels
    return pd.concat([labels, pd.DataFrame(rows)], ignore_index=True)
