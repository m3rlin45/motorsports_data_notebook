"""The solver in numpy, with the laps as a batch axis.

Every array here has one entry per lap of the batch: a step along the
track is taken by all the laps at once, and so is a step of the search
for u. A lap of a batch is driven exactly as it would be alone (the same
arithmetic in the same order), which takes three things a naive batch
does not have:

- each lap is turned to start at its own slowest apex;
- a forward step that does not settle is bisected for the laps it does
  not settle on, each until its own bracket is closed;
- a lap that has closed (its speed at the start line repeats) is left
  alone while the others go round again.

This is the reference the compiled lanes of :mod:`._kernel` are held to,
and what runs where numba is not installed. It takes and returns what
:func:`._kernel.solve_lanes` and :func:`._kernel.drive_lanes` do. Along
the track it is a loop in Python over the nodes, so it pays from a few
hundred laps at once; one lap alone costs about 200 µs a node.
"""

from __future__ import annotations

import numpy as np

from ._kernel import (
    BRENT_MAXITER,
    BRENT_RTOL,
    CLOSE_ROUNDS,
    CLOSE_TOL,
    DS,
    FLOOR,
    INV_DV,
    NO_ROOT,
    OK,
    ROUNDS,
    STEP_TOL,
    U_MAX,
    U_MIN,
    U_TOL,
    V_MAX,
    G,
)


class _Lanes:
    """The laps being driven, node first: row ``i`` of an array holds node
    ``i`` of every lap (already turned to the lap's own start)."""

    def __init__(
        self,
        kap: np.ndarray,
        g_cos: np.ndarray,
        kv: np.ndarray,
        g_grd: np.ndarray,
        g_sb: np.ndarray,
        lim: np.ndarray,
        ptabs: np.ndarray,
        which: np.ndarray,
        mu: np.ndarray,
        mub: np.ndarray,
        c: np.ndarray,
        kc: np.ndarray,
        be: np.ndarray,
        cst: np.ndarray,
    ) -> None:
        self.kap, self.g_cos, self.kv, self.g_grd, self.lim = kap, g_cos, kv, g_grd, lim
        self.g_sb = g_sb
        self.ptabs, self.which = ptabs, which
        self.table = ptabs[0] if len(ptabs) == 1 else None
        self.mu, self.mub, self.c, self.kc, self.be = mu, mub, c, kc, be
        self.curved = bool(np.any(be > 0.0))
        self.cst = cst
        self.g = float(cst[G])
        self.ds = float(cst[DS])
        self.inv_dv = float(cst[INV_DV])
        self.floor = float(cst[FLOOR])
        self.floor2 = self.floor * self.floor
        self.tol = float(cst[STEP_TOL])
        self.rounds = int(cst[ROUNDS])

    def take(self, keep: np.ndarray) -> "_Lanes":
        """The laps ``keep`` alone."""
        return _Lanes(
            np.ascontiguousarray(self.kap[:, keep]),
            np.ascontiguousarray(self.g_cos[:, keep]),
            np.ascontiguousarray(self.kv[:, keep]),
            np.ascontiguousarray(self.g_grd[:, keep]),
            np.ascontiguousarray(self.g_sb[:, keep]),
            np.ascontiguousarray(self.lim[:, keep]),
            self.ptabs,
            self.which[keep],
            self.mu[keep],
            self.mub[keep],
            self.c[keep],
            self.kc[keep],
            self.be[keep],
            self.cst,
        )

    def drive(self, v: np.ndarray, i: int, sel: np.ndarray | None = None) -> np.ndarray:
        """What accelerates the car at node ``i`` and speed ``v``, per lap
        (of the laps ``sel`` when given): the lesser of the engine and
        what is left of the grip, less drag and grade."""
        kap, g_cos, kv, g_grd = self.kap[i], self.g_cos[i], self.kv[i], self.g_grd[i]
        g_sb = self.g_sb[i]
        mu, c, kc, which = self.mu, self.c, self.kc, self.which
        if sel is not None:
            kap, g_cos, kv, g_grd = kap[sel], g_cos[sel], kv[sel], g_grd[sel]
            g_sb = g_sb[sel]
            mu, c, kc, which = mu[sel], c[sel], kc[sel], which[sel]
        lat = v * v * kap - g_sb  # what the tyres hold: cornering less the bank's share
        a_y = mu * (g_cos + v * v * kv)
        room = a_y * a_y - lat * lat
        x = v * self.inv_dv
        k = x.astype(np.int64)
        if self.table is not None:
            low, high = self.table[k], self.table[k + 1]
            share = self.table[-1]
        else:
            low, high = self.ptabs[which, k], self.ptabs[which, k + 1]
            share = self.ptabs[which, -1]
        p = low + (x - k) * (high - low)
        engine = p / v - kc * lat * lat / self.g
        # the driven axle's share of what is left of the grip: the table's last entry
        grip = share * np.sqrt(np.where(room > 0.0, room, 0.0))
        return np.asarray(np.where(engine < grip, engine, grip) - c * v * v - g_grd)

    def brake(self, v: np.ndarray, i: int) -> np.ndarray:
        """The braking at node ``i`` and speed ``v``, per lap."""
        load = self.g_cos[i] + v * v * self.kv[i]
        a_y = self.mu * load
        with np.errstate(divide="ignore", invalid="ignore"):
            # of the grip, by cornering
            used = np.where(a_y > 0.0, np.abs(v * v * self.kap[i] - self.g_sb[i]) / a_y, 1.0)
            left = 1.0 - used
            if self.curved:
                # between the diamond and the ellipse, each lap its own share of the way
                round_ = np.sqrt(np.where(left > 0.0, 1.0 - used * used, 0.0))
                bent = (1.0 - self.be) * left + self.be * round_
                left = np.where((left > 0.0) & (self.be > 0.0), bent, left)
        left = np.where(left > 0.0, left, 0.0)
        return np.asarray(-self.mub * load * left - self.c * v * v - self.g_grd[i])

    def step_on(self, v_i: np.ndarray, i: int, j: int) -> np.ndarray:
        """The speeds at node ``j`` from those at ``i``, accelerating."""
        e_i = v_i * v_i
        f_i = self.drive(v_i, i)
        v_j = v_i
        moved = np.zeros(len(v_i))
        for _ in range(self.rounds):
            e_j = e_i + self.ds * (f_i + self.drive(v_j, j))
            new = np.sqrt(np.where(e_j > self.floor2, e_j, self.floor2))
            moved = np.abs(new - v_j)
            v_j = new
        bad = np.flatnonzero(~(moved < self.tol))
        if bad.size:
            # At the edge of the envelope the iteration hops across the
            # root; those laps are bisected, each until its own bracket
            # is closed.
            e_b, f_b = e_i[bad], f_i[bad]
            lo = np.full(bad.size, self.floor)
            hi = np.maximum(v_i[bad], self.lim[j][bad]) + 5.0
            wide = hi - lo > self.tol
            while wide.any():
                mid = 0.5 * (lo + hi)
                over = mid * mid - e_b - self.ds * (f_b + self.drive(mid, j, bad)) > 0.0
                hi = np.where(wide & over, mid, hi)
                lo = np.where(wide & ~over, mid, lo)
                wide = hi - lo > self.tol
            v_j = v_j.copy()
            v_j[bad] = 0.5 * (lo + hi)
        return v_j

    def step_back(self, v_j: np.ndarray, i: int, j: int) -> np.ndarray:
        """The speeds at node ``i`` from which braking reaches ``v_j`` at ``j``."""
        e_j = v_j * v_j
        b_j = self.brake(v_j, j)
        lim = self.lim[i]
        v_i = v_j
        for _ in range(self.rounds):
            e_i = e_j - self.ds * (b_j + self.brake(np.where(v_i < lim, v_i, lim), i))
            v_i = np.sqrt(np.where(e_i > self.floor2, e_i, self.floor2))
        return v_i


def open_peaks(v: np.ndarray, ds: float, tau: np.ndarray, closed: bool) -> np.ndarray:
    """:func:`._kernel.open_peaks` for every lap at once: ``v`` is nodes ×
    laps and ``tau`` each lap's own time (a lap with none is left as it
    is). Every lap's windows advance a node at a time together; a lap
    whose window has closed stands still."""
    n, laps = v.shape
    steps = n if closed else n - 1
    lane = np.arange(laps)[None, :]
    dt = ds / (0.5 * (v + np.roll(v, -1, axis=0)))  # from each node to the next
    want = np.broadcast_to(tau > 0.0, (n, laps))

    first = np.minimum.accumulate(v, axis=0)  # the lowest from an open profile's start

    def sweep(values: np.ndarray, step: int, lowest: bool) -> np.ndarray:
        m = values.copy()
        acc = np.zeros((n, laps))
        k = np.broadcast_to(np.arange(n)[:, None], (n, laps)).copy()
        live = want.copy()
        for _ in range(steps):
            if not closed:
                edge = live & (k == (n - 1 if step > 0 else 0))
                if step < 0:  # a window reaching before the start holds only what is in it
                    m = np.where(edge & (first > m), first, m)
                live = live & ~edge
            if not live.any():
                break
            j = (k + step) % n
            span = dt[k, lane] if step > 0 else dt[j, lane]
            here, there = values[k, lane], values[j, lane]
            whole = live & (acc + span <= tau)
            part = live & ~whole
            with np.errstate(invalid="ignore", divide="ignore"):
                end = here + (tau - acc) / span * (there - here)
            better = (there < m) if lowest else (there > m)
            m = np.where(whole & better, there, m)
            better = (end < m) if lowest else (end > m)
            m = np.where(part & better, end, m)
            acc = np.where(whole, acc + span, acc)
            k = np.where(whole, j, k)
            live = whole
        return m

    low = sweep(v, 1, True)
    top = sweep(low, -1, False)
    return np.asarray(np.where(top < v, top, v))


def passes(
    kap: np.ndarray,
    grd: np.ndarray,
    kv: np.ndarray,
    cos: np.ndarray,
    gsb: np.ndarray,
    ptabs: np.ndarray,
    which: np.ndarray,
    mu: np.ndarray,
    mub: np.ndarray,
    c: np.ndarray,
    kc: np.ndarray,
    be: np.ndarray,
    ons: np.ndarray,
    cst: np.ndarray,
    *,
    closed: bool = True,
    v_start: float = float("nan"),
    v_end: float = float("nan"),
) -> np.ndarray:
    """Node speeds of every lap, ``(laps, nodes)`` in the profile's order.

    ``kap``, ``grd``, ``kv``, ``cos`` and ``gsb`` are the track's, as
    :func:`.solver._road` makes them (see :func:`._kernel.passes`); ``mu``
    and ``mub`` each lap's grip at its own u, ``c`` and ``kc`` its drag and
    cornering drag, ``be`` how far its braking limit is from the diamond
    towards the ellipse, ``ons`` the time it coasts between drive and
    braking, ``which`` the row of ``ptabs`` its engine reads.
    """
    n, laps = len(kap), len(mu)
    g, v_max = float(cst[G]), float(cst[V_MAX])
    floor2 = float(cst[FLOOR]) * float(cst[FLOOR])
    den = kap[None, :] - mu[:, None] * kv[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        held = (mu[:, None] * g * cos[None, :] + gsb[None, :]) / np.maximum(den, 1e-9)
        v2 = np.where(den > 1e-9, held, v_max**2)
    # a road leaning out by more than the grip holds leaves the floor
    lim = np.minimum(np.sqrt(np.where(v2 < floor2, floor2, v2)), v_max)
    # each lap from its own slowest apex
    start = lim.argmin(axis=1) if closed else np.zeros(laps, dtype=np.int64)
    idx = (np.arange(n)[None, :] + start[:, None]) % n

    def turned(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a[idx].T)

    lanes = _Lanes(
        turned(kap),
        turned(g * cos),
        turned(kv),
        turned(g * grd),
        turned(gsb),
        np.ascontiguousarray(np.take_along_axis(lim, idx, axis=1).T),
        ptabs,
        which,
        mu,
        mub,
        c,
        kc,
        be,
        cst,
    )
    v = lanes.lim.copy()
    if closed:
        live, on = lanes, np.arange(laps)
        for _ in range(int(cst[CLOSE_ROUNDS])):
            w = v[:, on]
            for i in range(n - 1):
                w[i + 1] = np.minimum(live.lim[i + 1], live.step_on(w[i], i, i + 1))
            closing = np.minimum(live.lim[0], live.step_on(w[n - 1], n - 1, 0))
            gap = w[0] - closing
            w[0] = closing
            v[:, on] = w
            again = ~(np.abs(gap) < cst[CLOSE_TOL])
            if not again.any():
                break
            if not again.all():  # the laps that have closed are left as they are
                on = on[again]
                live = lanes.take(on)
        # backward, twice round so that a braking zone across the start is whole
        for _ in range(2):
            v[n - 1] = np.minimum(v[n - 1], lanes.step_back(v[0], n - 1, 0))
            for i in range(n - 2, -1, -1):
                v[i] = np.minimum(v[i], lanes.step_back(v[i + 1], i, i + 1))
    else:
        if not np.isnan(v_start):
            v[0] = np.minimum(lanes.lim[0], v_start)
        for i in range(n - 1):
            v[i + 1] = np.minimum(lanes.lim[i + 1], lanes.step_on(v[i], i, i + 1))
        if not np.isnan(v_end):
            v[n - 1] = np.minimum(v[n - 1], v_end)
        for i in range(n - 2, -1, -1):
            v[i] = np.minimum(v[i], lanes.step_back(v[i + 1], i, i + 1))
    if np.any(ons > 0.0):
        v = open_peaks(v, float(cst[DS]), ons, closed)
    out = np.empty((laps, n))
    out[np.arange(laps)[:, None], idx] = v.T
    return out


def lap_times(v: np.ndarray, ds: float, closed: bool = True) -> np.ndarray:
    """The lap time of each row of node speeds, summed along the lap in
    order (as the compiled lanes sum it, so the two agree to the last bit)."""
    if v.shape[1] < 2:
        return np.zeros(len(v))
    nxt = np.roll(v, -1, axis=1) if closed else v[:, 1:]
    first = v if closed else v[:, :-1]
    return np.asarray(np.cumsum(ds / (0.5 * (first + nxt)), axis=1)[:, -1])


def drive_lanes(
    u: np.ndarray,
    kap: np.ndarray,
    grd: np.ndarray,
    kv: np.ndarray,
    cos: np.ndarray,
    gsb: np.ndarray,
    ptabs: np.ndarray,
    which: np.ndarray,
    mu_y: np.ndarray,
    mu_b: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    kc: np.ndarray,
    be: np.ndarray,
    ons: np.ndarray,
    cst: np.ndarray,
    *,
    closed: bool = True,
    v_start: float = float("nan"),
    v_end: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """Lap time and node speeds of every lap at its own ``u``."""
    v = passes(
        kap, grd, kv, cos, gsb, ptabs, which, mu_y * u * w, mu_b * u * w, c, kc, be, ons, cst,
        closed=closed, v_start=v_start, v_end=v_end,
    )  # fmt: skip
    return lap_times(v, float(cst[DS]), closed), v


def solve_lanes(
    target: np.ndarray,
    guess: np.ndarray,
    kap: np.ndarray,
    grd: np.ndarray,
    kv: np.ndarray,
    cos: np.ndarray,
    gsb: np.ndarray,
    ptabs: np.ndarray,
    which: np.ndarray,
    mu_y: np.ndarray,
    mu_b: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    kc: np.ndarray,
    be: np.ndarray,
    ons: np.ndarray,
    cst: np.ndarray,
    *,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The u at which each lap is done in its ``target`` time, its status
    and its node speeds at that u (NaN without a root).

    :func:`.solver.solve_u_near` for every lap at once: the bracket around
    the guess, its widening and Brent's method all advance one step per
    pass over the track, each lap by its own rule, and a lap that is done
    drops out of the passes that follow.
    """
    laps, n = len(target), len(kap)
    u_min, u_max, u_tol = float(cst[U_MIN]), float(cst[U_MAX]), float(cst[U_TOL])
    u = np.full(laps, np.nan)
    status = np.full(laps, NO_ROOT, dtype=np.int64)

    def miss(us: np.ndarray, at: np.ndarray) -> np.ndarray:
        if not at.size:
            return np.empty(0)
        t, _ = drive_lanes(
            us, kap, grd, kv, cos, gsb, ptabs, which[at], mu_y[at], mu_b[at], w[at], c[at], kc[at],
            be[at], ons[at], cst, closed=closed,
        )  # fmt: skip
        return np.asarray(t - target[at])

    def whole_range(at: np.ndarray) -> None:
        """:func:`.solver.solve_u`: bisection over the whole range of u."""
        f_lo = miss(np.full(at.size, u_min), at)
        f_hi = miss(np.full(at.size, u_max), at)
        at = at[(f_hi <= 0.0) & (0.0 <= f_lo)]
        lo, hi = np.full(at.size, u_min), np.full(at.size, u_max)
        while at.size and hi[0] - lo[0] > u_tol:
            mid = 0.5 * (lo + hi)
            slow = miss(mid, at) > 0.0
            lo, hi = np.where(slow, mid, lo), np.where(slow, hi, mid)
        u[at] = 0.5 * (lo + hi)
        status[at] = OK

    finite = np.isfinite(target) & np.isfinite(guess)
    at = np.flatnonzero(finite)
    lo = np.maximum(guess[at] * 0.94, u_min)
    hi = np.minimum(guess[at] * 1.06, u_max)
    f_lo, f_hi = miss(lo, at), miss(hi, at)
    looking = np.ones(at.size, dtype=bool)
    for _ in range(8):
        looking &= ~((f_lo >= 0.0) & (0.0 >= f_hi))
        slow = looking & (f_lo < 0.0)  # slower than the bracket's slow end
        fast = looking & ~(f_lo < 0.0)
        slow &= ~(lo <= u_min)  # nothing beyond the range: no root
        fast &= ~(hi >= u_max)
        looking = slow | fast
        if not looking.any():
            break
        hi[slow], f_hi[slow] = lo[slow], f_lo[slow]
        lo[slow] = np.maximum(lo[slow] * 0.85, u_min)
        lo[fast], f_lo[fast] = hi[fast], f_hi[fast]
        hi[fast] = np.minimum(hi[fast] * 1.15, u_max)
        got = miss(np.where(slow, lo, hi)[looking], at[looking])
        f_lo[slow] = got[slow[looking]]
        f_hi[fast] = got[fast[looking]]
    held = (f_lo >= 0.0) & (0.0 >= f_hi)
    # still looking after eight widenings: over the whole range, like a
    # lap without a finite time or guess
    whole_range(np.concatenate([np.flatnonzero(~finite), at[looking & ~held]]))
    at_lo = held & (f_lo == 0.0)
    at_hi = held & ~at_lo & (f_hi == 0.0)
    u[at[at_lo]], u[at[at_hi]] = lo[at_lo], hi[at_hi]
    status[at[held]] = OK

    # Brent's method (scipy's brentq), every lap by its own branch.
    go = held & ~at_lo & ~at_hi
    at = at[go]
    xpre, xcur, fpre, fcur = lo[go], hi[go], f_lo[go], f_hi[go]
    xblk, fblk = np.zeros(at.size), np.zeros(at.size)
    spre, scur = np.zeros(at.size), np.zeros(at.size)
    for _ in range(BRENT_MAXITER):
        if not at.size:
            break
        cross = (fpre != 0.0) & (fcur != 0.0) & ((fpre < 0.0) != (fcur < 0.0))
        xblk, fblk = np.where(cross, xpre, xblk), np.where(cross, fpre, fblk)
        spre = np.where(cross, xcur - xpre, spre)
        scur = np.where(cross, spre, scur)
        swap = np.abs(fblk) < np.abs(fcur)
        xpre, xcur, xblk = (
            np.where(swap, xcur, xpre),
            np.where(swap, xblk, xcur),
            np.where(swap, xcur, xblk),
        )
        fpre, fcur, fblk = (
            np.where(swap, fcur, fpre),
            np.where(swap, fblk, fcur),
            np.where(swap, fcur, fblk),
        )
        delta = (u_tol + BRENT_RTOL * np.abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        done = (fcur == 0.0) | (np.abs(sbis) < delta)
        u[at[done]] = xcur[done]
        if done.any():
            keep = ~done
            at = at[keep]
            xpre, xcur, xblk = xpre[keep], xcur[keep], xblk[keep]
            fpre, fcur, fblk = fpre[keep], fcur[keep], fblk[keep]
            spre, scur, delta, sbis = spre[keep], scur[keep], delta[keep], sbis[keep]
            if not at.size:
                break
        with np.errstate(divide="ignore", invalid="ignore"):
            inter = -fcur * (xcur - xpre) / (fcur - fpre)
            dpre = (fpre - fcur) / (xpre - xcur)
            dblk = (fblk - fcur) / (xblk - xcur)
            extra = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
        stry = np.where(xpre == xblk, inter, extra)
        bound = 3.0 * np.abs(sbis) - delta
        bound = np.where(np.abs(spre) < bound, np.abs(spre), bound)
        try_it = (np.abs(spre) > delta) & (np.abs(fcur) < np.abs(fpre))
        take = try_it & (2.0 * np.abs(stry) < bound)
        spre, scur = np.where(take, scur, sbis), np.where(take, stry, sbis)
        xpre, fpre = xcur, fcur
        xcur = xcur + np.where(np.abs(scur) > delta, scur, np.where(sbis > 0.0, delta, -delta))
        fcur = miss(xcur, at)
    u[at] = xcur

    v = np.full((laps, n), np.nan)
    ok = np.flatnonzero(status == OK)
    if ok.size:
        _, v[ok] = drive_lanes(
            u[ok], kap, grd, kv, cos, gsb, ptabs, which[ok], mu_y[ok], mu_b[ok], w[ok], c[ok], kc[ok],
            be[ok], ons[ok], cst, closed=closed,
        )  # fmt: skip
    return u, status, v


__all__ = ["drive_lanes", "lap_times", "passes", "solve_lanes"]
