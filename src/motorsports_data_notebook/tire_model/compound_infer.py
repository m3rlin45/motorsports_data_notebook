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

_AXLE_CORNERS = {"front": ("fl", "fr"), "rear": ("rl", "rr")}
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
) -> tuple[dict[tuple[str, str, str, str], tuple[float, float, float]], list[AxleAssignment]]:
    """Joint EM over compound assignments and per-compound K.

    Returns ``(k_by_compound, assignments)`` where ``k_by_compound`` maps
    ``(car, compound, corner, condition) -> (K, stderr, n_effective)`` and
    ``assignments`` carries the final per-(session, axle) posteriors.
    """
    if laps_for_fit.empty or labels.empty:
        return {}, []

    stats = _suff_stats(laps_for_fit, tau_by_car_corner_cond, c_track_by_track)
    if stats.empty:
        return {}, []
    stats["axle"] = stats["corner"].map(_CORNER_AXLE)

    label_cols = {"front": "compound_front", "rear": "compound_rear"}
    pinned: dict[tuple[str, str], str] = {}
    for r in labels.itertuples():
        for axle, col in label_cols.items():
            comp = getattr(r, col)
            if isinstance(comp, str) and comp:
                pinned[(str(r.session_id), axle)] = comp

    out_k: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    assignments: list[AxleAssignment] = []

    for car_key, car_stats in stats.groupby("car"):
        car = str(car_key)
        # Eligibility: ≥2 compounds with ≥MIN_LABELED_SESSIONS pinned sessions.
        pinned_here: dict[str, set] = {}
        for (sid, axle), comp in pinned.items():
            if sid in set(car_stats["session_id"]):
                pinned_here.setdefault(comp, set()).add(sid)
        compounds = sorted(
            c for c, sids in pinned_here.items() if len(sids) >= MIN_LABELED_SESSIONS
        )
        if len(compounds) < 2:
            # Still fit K for whatever labeled compounds exist (no latents).
            compounds = sorted(pinned_here)
            if not compounds:
                continue
            fixed_only = True
        else:
            fixed_only = False

        # Units: (session, axle) with enough laps.
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

        # Responsibilities: pinned one-hot; unlabeled start uniform.
        resp = np.full((n_units, n_comp), 1.0 / n_comp)
        pin_mask = np.zeros(n_units, dtype=bool)
        for u, i in unit_idx.items():
            comp = pinned.get(u)
            if comp is not None and comp in comp_idx:
                resp[i] = 0.0
                resp[i, comp_idx[comp]] = 1.0
                pin_mask[i] = True
            elif fixed_only:
                # No latents for this car: unlabeled units don't participate.
                resp[i] = 0.0

        # Index sufficient stats by unit.
        s_unit = car_stats.copy()
        s_unit["unit"] = list(zip(s_unit["session_id"], s_unit["axle"]))
        s_unit = s_unit[s_unit["unit"].isin(unit_idx)]
        s_unit["ui"] = s_unit["unit"].map(unit_idx)

        buckets = s_unit.groupby(["corner", "condition"])
        sigma2: dict[str, float] = {c: 25.0 for c in _CORNER_AXLE}  # start: (5 °C)²
        k_val: dict[tuple[str, str, str], float] = {}  # (compound, corner, cond) -> K

        def k_for_scoring(comp: str, corner: str, cond: str) -> float | None:
            """K for likelihood scoring: exact condition, then dry, then any.

            A compound with NO K anywhere for this corner must not be
            scored at all — treating a missing bucket as zero residual
            would make "no coverage" look like a perfect fit.
            """
            for c2 in (cond, "dry"):
                k = k_val.get((comp, corner, c2))
                if k is not None:
                    return k
            for (c_comp, c_corner, _c2), k in k_val.items():
                if c_comp == comp and c_corner == corner:
                    return k
            return None

        prev_ll = -np.inf
        for _ in range(max_iter):
            # ---- M-step: responsibility-weighted least squares per bucket ----
            k_val.clear()
            for bkey, grp in buckets:
                corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
                ui = grp["ui"].to_numpy()
                for comp, j in comp_idx.items():
                    w = resp[ui, j]
                    w_pinned = float(np.sum(w * pin_mask[ui] * grp["n"].to_numpy()))
                    if not fixed_only and w_pinned < MIN_PINNED_LAPS_PER_BUCKET:
                        continue
                    sxx = float(np.sum(w * grp["sxx"].to_numpy()))
                    if sxx <= 0:
                        continue
                    sxy = float(np.sum(w * grp["sxy"].to_numpy()))
                    k_val[(comp, corner, cond)] = sxy / sxx
            # Residual variance per corner (pooled over compounds/conds).
            for corner in _CORNER_AXLE:
                num = den = 0.0
                for bkey, grp in buckets:
                    c2, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
                    if c2 != corner:
                        continue
                    ui = grp["ui"].to_numpy()
                    for comp, j in comp_idx.items():
                        k = k_val.get((comp, corner, cond))
                        if k is None:
                            continue
                        w = resp[ui, j]
                        rss_arr = (
                            grp["syy"].to_numpy()
                            - 2 * k * grp["sxy"].to_numpy()
                            + k * k * grp["sxx"].to_numpy()
                        )
                        num += float(np.sum(w * rss_arr))
                        den += float(np.sum(w * grp["n"].to_numpy()))
                if den > 0:
                    sigma2[corner] = max(num / den, SIGMA2_FLOOR)

            # ---- E-step: posteriors for un-pinned units ----
            loglik = np.zeros((n_units, n_comp))
            covered = np.zeros((n_units, n_comp))
            for bkey, grp in buckets:
                corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
                ui = grp["ui"].to_numpy()
                s2 = sigma2[corner]
                for comp, j in comp_idx.items():
                    k = k_for_scoring(comp, corner, cond)
                    if k is None:
                        continue
                    rss_arr = (
                        grp["syy"].to_numpy()
                        - 2 * k * grp["sxy"].to_numpy()
                        + k * k * grp["sxx"].to_numpy()
                    )
                    np.add.at(loglik[:, j], ui, -0.5 * rss_arr / s2)
                    np.add.at(covered[:, j], ui, grp["n"].to_numpy())
            # A compound that covers none of a unit's laps is not a candidate.
            loglik = np.where(covered > 0, loglik, -np.inf)
            free = ~pin_mask if not fixed_only else np.zeros(n_units, dtype=bool)
            if free.any():
                n_u = np.array([max(float(unit_lap_counts.get(u, 1)), 1.0) for u in units])
                temper = np.minimum(1.0, SESSION_EFF_SAMPLES / n_u)[:, None]
                ll = (loglik * temper)[free]
                row_max = ll.max(axis=1, keepdims=True)
                ok_rows = np.isfinite(row_max[:, 0])
                p = np.zeros_like(ll)
                if ok_rows.any():
                    z = np.exp(ll[ok_rows] - row_max[ok_rows])
                    p[ok_rows] = z / z.sum(axis=1, keepdims=True)
                resp[free] = p
            total_ll = float(np.sum(np.where(resp > 0, loglik * resp, 0.0)))
            if abs(total_ll - prev_ll) < EM_TOL * (1 + abs(prev_ll)):
                break
            prev_ll = total_ll

        # Outlier gate: free units whose best cluster still fits badly get
        # their responsibilities zeroed (unknown compound — stays unlabeled).
        # The gate's variance comes from PINNED sessions only: the mixture
        # σ² is inflated by the very outliers we are trying to detect.
        if not fixed_only:
            sig_num: dict[str, float] = {c: 0.0 for c in _CORNER_AXLE}
            sig_den: dict[str, float] = {c: 0.0 for c in _CORNER_AXLE}
            chi2_num = np.zeros((n_units, n_comp))
            for bkey, grp in buckets:
                corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
                ui = grp["ui"].to_numpy()
                for comp, j in comp_idx.items():
                    k = k_for_scoring(comp, corner, cond)
                    if k is None:
                        np.add.at(chi2_num[:, j], ui, np.inf * np.ones(len(ui)))
                        continue
                    rss_arr = (
                        grp["syy"].to_numpy()
                        - 2 * k * grp["sxy"].to_numpy()
                        + k * k * grp["sxx"].to_numpy()
                    )
                    np.add.at(chi2_num[:, j], ui, rss_arr)
                    w_pin = resp[ui, j] * pin_mask[ui]
                    sig_num[corner] += float(np.sum(w_pin * rss_arr))
                    sig_den[corner] += float(np.sum(w_pin * grp["n"].to_numpy()))
            sigma2_clean = np.mean(
                [
                    max(sig_num[c] / sig_den[c], SIGMA2_FLOOR) if sig_den[c] > 0 else SIGMA2_FLOOR
                    for c in _CORNER_AXLE
                ]
            )
            n_u_arr = np.array([max(float(unit_lap_counts.get(u, 1)), 1.0) for u in units])
            best_chi2 = chi2_num.min(axis=1) / (n_u_arr * sigma2_clean)
            outlier = (~pin_mask) & (best_chi2 > OUTLIER_CHI2_PER_LAP)
            if outlier.any():
                resp[outlier] = 0.0
                # One more M-step so exported K excludes the outliers.
                k_val.clear()
                for bkey, grp in buckets:
                    corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
                    ui = grp["ui"].to_numpy()
                    for comp, j in comp_idx.items():
                        w = resp[ui, j]
                        w_pinned = float(np.sum(w * pin_mask[ui] * grp["n"].to_numpy()))
                        if not fixed_only and w_pinned < MIN_PINNED_LAPS_PER_BUCKET:
                            continue
                        sxx = float(np.sum(w * grp["sxx"].to_numpy()))
                        if sxx > 0:
                            k_val[(comp, corner, cond)] = float(
                                np.sum(w * grp["sxy"].to_numpy()) / sxx
                            )

        # ---- Export K with stderr + effective n ----
        for bkey, grp in buckets:
            corner, cond = str(bkey[0]), str(bkey[1])  # type: ignore[index]
            ui = grp["ui"].to_numpy()
            for comp, j in comp_idx.items():
                k = k_val.get((comp, corner, cond))
                if k is None:
                    continue
                w = resp[ui, j]
                sxx = float(np.sum(w * grp["sxx"].to_numpy()))
                n_eff = float(np.sum(w * grp["n"].to_numpy()))
                if sxx <= 0 or n_eff < MIN_LAPS_PER_AXLE:
                    continue
                rss = float(
                    np.sum(
                        w
                        * (
                            grp["syy"].to_numpy()
                            - 2 * k * grp["sxy"].to_numpy()
                            + k * k * grp["sxx"].to_numpy()
                        )
                    )
                )
                stderr = float(np.sqrt(max(rss, 0.0) / max(n_eff - 1, 1.0) / sxx))
                out_k[(str(car), comp, corner, cond)] = (k, stderr, n_eff)

        for u, i in unit_idx.items():
            j = int(np.argmax(resp[i]))
            if resp[i].sum() == 0:
                continue
            assignments.append(
                AxleAssignment(
                    session_id=u[0],
                    car=str(car),
                    axle=u[1],
                    compound=compounds[j],
                    responsibility=round(float(resp[i, j]), 3),
                    n_laps=unit_lap_counts.get(u, 0),
                    pinned=bool(pin_mask[i]),
                )
            )

    return out_k, assignments


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
            rows.append(
                {
                    "session_id": sid,
                    "compound_front": compound,
                    "compound_rear": compound,
                    "source": "seed",
                }
            )
    if not rows:
        return labels
    return pd.concat([labels, pd.DataFrame(rows)], ignore_index=True)
