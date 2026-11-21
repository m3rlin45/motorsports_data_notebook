"""The solver compiled, where numba is installed: one lane per lap.

The statements of :mod:`._tensor` (numpy, the laps as an axis) written
for one lap at a time so that numba can compile them: the same
trapezoidal steps in ``V²``, the same fixed-point rounds and the same
bisection where they do not settle, the same search for u, in the same
order of arithmetic. A batch is the lanes side by side on the machine's
cores (:func:`solve_lanes_parallel`, :func:`drive_lanes_parallel`). Tests
hold the two forms to the same numbers; without numba this module's
functions are plain Python and the solver uses :mod:`._tensor` instead.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - which branch runs depends on the environment
    from numba import njit, prange  # type: ignore[import-not-found,unused-ignore]

    HAVE_NUMBA = True
except Exception:  # noqa: BLE001 - any failure to import means: not compiled
    HAVE_NUMBA = False
    prange = range  # type: ignore[assignment,misc,unused-ignore]

    def njit(*args, **kwargs):  # type: ignore[no-untyped-def,no-redef]
        """A stand-in that leaves the function as it is."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


@njit(cache=True)
def _drive(v, i, kap, cos, kv, grd, gsb, ptab, inv_dv, mu, c, kc, g):  # type: ignore[no-untyped-def]
    lat = v * v * kap[i] - gsb[i]  # what the tyres hold: cornering less the bank's share
    a_y = mu * (g * cos[i] + v * v * kv[i])
    room = a_y * a_y - lat * lat
    x = v * inv_dv
    k = int(x)
    p = ptab[k] + (x - k) * (ptab[k + 1] - ptab[k])
    engine = p / v - kc * lat * lat / g
    grip = math.sqrt(room) if room > 0.0 else 0.0
    tyre = engine if engine < grip else grip
    return tyre - c * v * v - g * grd[i]


@njit(cache=True)
def _brake(v, i, kap, cos, kv, grd, gsb, mu, mub, c, g, be):  # type: ignore[no-untyped-def]
    load = g * cos[i] + v * v * kv[i]
    a_y = mu * load
    # of the grip, by cornering
    used = abs(v * v * kap[i] - gsb[i]) / a_y if a_y > 0.0 else 1.0
    left = 1.0 - used
    if left < 0.0:
        left = 0.0
    elif be > 0.0:
        # between the diamond (what the lateral leaves, taken linearly) and
        # the ellipse, ``be`` of the way
        left = (1.0 - be) * left + be * math.sqrt(1.0 - used * used)
    return -mub * load * left - c * v * v - g * grd[i]


@njit(cache=True)
def _step_on(  # type: ignore[no-untyped-def]
    v_i, i, j, kap, cos, kv, grd, gsb, lim, ptab, inv_dv, mu, c, kc, g, ds, floor, rounds, tol
):
    floor2 = floor * floor
    e_i = v_i * v_i
    f_i = _drive(v_i, i, kap, cos, kv, grd, gsb, ptab, inv_dv, mu, c, kc, g)
    v_j = v_i
    moved = 0.0
    for _ in range(rounds):
        e_j = e_i + ds * (f_i + _drive(v_j, j, kap, cos, kv, grd, gsb, ptab, inv_dv, mu, c, kc, g))
        new = math.sqrt(e_j if e_j > floor2 else floor2)
        moved = abs(new - v_j)
        v_j = new
    if moved < tol:
        return v_j
    lo = floor
    hi = (v_i if v_i > lim[j] else lim[j]) + 5.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        f_mid = _drive(mid, j, kap, cos, kv, grd, gsb, ptab, inv_dv, mu, c, kc, g)
        if mid * mid - e_i - ds * (f_i + f_mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _step_back(  # type: ignore[no-untyped-def]
    v_j, i, j, kap, cos, kv, grd, gsb, lim, mu, mub, c, g, be, ds, floor, rounds
):
    floor2 = floor * floor
    e_j = v_j * v_j
    b_j = _brake(v_j, j, kap, cos, kv, grd, gsb, mu, mub, c, g, be)
    v_i = v_j
    for _ in range(rounds):
        at = v_i if v_i < lim[i] else lim[i]
        e_i = e_j - ds * (b_j + _brake(at, i, kap, cos, kv, grd, gsb, mu, mub, c, g, be))
        v_i = math.sqrt(e_i if e_i > floor2 else floor2)
    return v_i


@njit(cache=True)
def passes(  # type: ignore[no-untyped-def]
    kap,
    grd,
    kv,
    cos,
    gsb,
    lim,
    ptab,
    inv_dv,
    mu,
    mub,
    c,
    kc,
    be,
    g,
    ds,
    closed,
    v_start,
    v_end,
    floor,
    rounds,
    tol,
    close_tol,
    close_rounds,
):
    """Node speeds from the arrays of a profile already turned to start at
    its slowest apex. ``v_start`` and ``v_end`` are NaN where none is
    given.

    The track's arrays are as :func:`.solver._road` makes them. On a flat
    road ``kap`` is ``|κ|``, ``kv`` the vertical curvature, ``cos`` the
    grade's cosine and ``gsb`` zero. On a road leaning into the turn by
    the angle b they are ``|κ|·cos b``, ``κ_v + |κ|·sin b``,
    ``cos θ·cos b`` and ``g·sin b``: the tyres then hold
    ``V²·kap − gsb`` sideways under the load ``g·cos + V²·kv``, and the
    statements are the flat road's with those two read so."""
    n = len(lim)
    v = lim.copy()
    if closed:
        for _ in range(close_rounds):
            for i in range(n - 1):
                step = _step_on(
                    v[i],
                    i,
                    i + 1,
                    kap,
                    cos,
                    kv,
                    grd,
                    gsb,
                    lim,
                    ptab,
                    inv_dv,
                    mu,
                    c,
                    kc,
                    g,
                    ds,
                    floor,
                    rounds,
                    tol,
                )
                v[i + 1] = step if step < lim[i + 1] else lim[i + 1]
            step = _step_on(
                v[n - 1],
                n - 1,
                0,
                kap,
                cos,
                kv,
                grd,
                gsb,
                lim,
                ptab,
                inv_dv,
                mu,
                c,
                kc,
                g,
                ds,
                floor,
                rounds,
                tol,
            )
            closing = step if step < lim[0] else lim[0]
            gap = v[0] - closing
            v[0] = closing
            if abs(gap) < close_tol:
                break
        for _ in range(2):
            back = _step_back(
                v[0], n - 1, 0, kap, cos, kv, grd, gsb, lim, mu, mub, c, g, be, ds, floor, rounds
            )
            if back < v[n - 1]:
                v[n - 1] = back
            for i in range(n - 2, -1, -1):
                back = _step_back(
                    v[i + 1],
                    i,
                    i + 1,
                    kap,
                    cos,
                    kv,
                    grd,
                    gsb,
                    lim,
                    mu,
                    mub,
                    c,
                    g,
                    be,
                    ds,
                    floor,
                    rounds,
                )
                if back < v[i]:
                    v[i] = back
    else:
        if not math.isnan(v_start):
            v[0] = v_start if v_start < lim[0] else lim[0]
        for i in range(n - 1):
            step = _step_on(
                v[i],
                i,
                i + 1,
                kap,
                cos,
                kv,
                grd,
                gsb,
                lim,
                ptab,
                inv_dv,
                mu,
                c,
                kc,
                g,
                ds,
                floor,
                rounds,
                tol,
            )
            v[i + 1] = step if step < lim[i + 1] else lim[i + 1]
        if not math.isnan(v_end) and v_end < v[n - 1]:
            v[n - 1] = v_end
        for i in range(n - 2, -1, -1):
            back = _step_back(
                v[i + 1],
                i,
                i + 1,
                kap,
                cos,
                kv,
                grd,
                gsb,
                lim,
                mu,
                mub,
                c,
                g,
                be,
                ds,
                floor,
                rounds,
            )
            if back < v[i]:
                v[i] = back
    return v


@njit(cache=True)
def open_peaks(v, ds, tau, closed, out):  # type: ignore[no-untyped-def]
    """``v`` with every speed peak cut flat where it is ``tau`` seconds
    long, into ``out``.

    Between full drive and full braking the car coasts for a while: the
    driver's foot goes from one pedal to the other and the brake builds.
    A lap that does this has a flat top ``tau`` long on each of its speed
    peaks: it leaves the drive curve early and joins the same braking
    curve. That is the opening of the speed trace by a window of ``tau``
    in time: the lowest speed in the ``tau`` after each node, then the
    highest of those in the ``tau`` before it. Minima are left as they
    are, and a peak already flatter than ``tau`` is too. A window's end
    falls between two nodes and is read linearly there, so the result
    moves smoothly with the lap. An open profile's ends are not peaks:
    a window reaching past either holds only what is inside.
    """
    n = len(v)
    steps = n if closed else n - 1
    dt = np.empty(n)  # from each node to the next
    for k in range(n):
        j = k + 1 if k + 1 < n else 0
        dt[k] = ds / (0.5 * (v[k] + v[j]))
    low = np.empty(n)
    for i in range(n):
        m = v[i]
        acc = 0.0
        k = i
        for _ in range(steps):
            if not closed and k == n - 1:
                break
            j = k + 1 if k + 1 < n else 0
            if acc + dt[k] <= tau:
                acc += dt[k]
                k = j
                if v[k] < m:
                    m = v[k]
            else:
                end = v[k] + (tau - acc) / dt[k] * (v[j] - v[k])
                if end < m:
                    m = end
                break
        low[i] = m
    first = v[0]  # the lowest speed from an open profile's start to each node
    for i in range(n):
        if v[i] < first:
            first = v[i]
        m = low[i]
        acc = 0.0
        k = i
        for _ in range(steps):
            if not closed and k == 0:
                # the windows that begin before an open profile's start hold
                # only what is in it: the start is not a peak to be cut
                if first > m:
                    m = first
                break
            j = k - 1 if k > 0 else n - 1
            if acc + dt[j] <= tau:
                acc += dt[j]
                k = j
                if low[k] > m:
                    m = low[k]
            else:
                end = low[k] + (tau - acc) / dt[j] * (low[j] - low[k])
                if end > m:
                    m = end
                break
        out[i] = m if m < v[i] else v[i]


# ------------------------------------------------- the track's own geometry


@njit(cache=True)
def nearest_vertex(xr, yr, x, y, centre, half, last):  # type: ignore[no-untyped-def]
    """For each point ``(x, y)`` the vertex of the line ``(xr, yr)`` nearest
    to it among those within ``half`` of ``centre`` (clipped to
    ``0..last``); the first of equals, as ``argmin`` has it."""
    out = np.empty(len(x), np.int64)
    for i in range(len(x)):
        first = centre[i] - half
        if first < 0:
            first = 0
        elif first > last:
            first = last
        best = first
        best_d = np.inf
        for k in range(-half, half + 1):
            c = centre[i] + k
            if c < 0:
                c = 0
            elif c > last:
                c = last
            dx = x[i] - xr[c]
            dy = y[i] - yr[c]
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best = c
        out[i] = best
    return out


@njit(cache=True)
def sorted_bounds(key, query, right):  # type: ignore[no-untyped-def]
    """``np.searchsorted(key, query, side)`` where both are sorted: one
    walk along the two instead of a binary search per query."""
    n = len(key)
    out = np.empty(len(query), np.int64)
    j = 0
    for i in range(len(query)):
        q = query[i]
        if right:
            while j < n and key[j] <= q:
                j += 1
        else:
            while j < n and key[j] < q:
                j += 1
        out[i] = j
    return out


# --------------------------------------------------------- a batch of laps
#
# A lane is one lap: its own car (grip, drag, power table), its own pace
# knob or its own lap time to be met. The lanes share the track and nothing
# else, so they run side by side on the machine's cores.

#: Where the constants sit in the ``cst`` array the batch functions take.
INV_DV, G, DS, V_MAX, FLOOR, STEP_TOL, CLOSE_TOL, ROUNDS, CLOSE_ROUNDS, U_MIN, U_MAX, U_TOL = range(
    12
)
#: Brent's method as scipy runs it: its default ``rtol`` and ``maxiter``.
BRENT_RTOL = 4.0 * 2.220446049250313e-16
BRENT_MAXITER = 100
OK, NO_ROOT = 0, 1


@njit(cache=True)
def lap_nodes(kap, grd, kv, cos, gsb, ptab, mu, mub, c, kc, be, ons, cst, out):  # type: ignore[no-untyped-def]
    """Node speeds of one closed lap into ``out`` (in the profile's own
    order) and its lap time: the cornering limit, the turn to the slowest
    apex and :func:`passes`, as :func:`.solver._passes` does them."""
    n = len(kap)
    g = cst[G]
    v_max = cst[V_MAX]
    floor2 = cst[FLOOR] * cst[FLOOR]
    lim = np.empty(n)
    start = 0
    for i in range(n):
        den = kap[i] - mu * kv[i]
        v2 = (mu * g * cos[i] + gsb[i]) / den if den > 1e-9 else v_max * v_max
        if v2 < floor2:  # a road leaning out by more than the grip holds
            v2 = floor2
        top = math.sqrt(v2)
        if top > v_max:
            top = v_max
        lim[i] = top
        if top < lim[start]:
            start = i
    kap_r = np.empty(n)
    grd_r = np.empty(n)
    kv_r = np.empty(n)
    cos_r = np.empty(n)
    gsb_r = np.empty(n)
    lim_r = np.empty(n)
    for i in range(n):
        k = i + start
        if k >= n:
            k -= n
        kap_r[i] = kap[k]
        grd_r[i] = grd[k]
        kv_r[i] = kv[k]
        cos_r[i] = cos[k]
        gsb_r[i] = gsb[k]
        lim_r[i] = lim[k]
    v = passes(
        kap_r, grd_r, kv_r, cos_r, gsb_r, lim_r, ptab, cst[INV_DV], mu, mub, c, kc, be, g, cst[DS],
        True, np.nan, np.nan, cst[FLOOR], int(cst[ROUNDS]), cst[STEP_TOL], cst[CLOSE_TOL],
        int(cst[CLOSE_ROUNDS]),
    )  # fmt: skip
    if ons > 0.0:
        flat = np.empty(n)
        open_peaks(v, cst[DS], ons, True, flat)
        v = flat
    for i in range(n):
        k = i + start
        if k >= n:
            k -= n
        out[k] = v[i]
    t = 0.0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        t += cst[DS] / (0.5 * (out[i] + out[j]))
    return t


@njit(cache=True)
def _miss(u, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out):  # type: ignore[no-untyped-def]
    return (
        lap_nodes(
            kap, grd, kv, cos, gsb, ptab, mu_y * u * w, mu_b * u * w, c, kc, be, ons, cst, out
        )
        - target
    )


@njit(cache=True)
def _bisect_u(target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out):  # type: ignore[no-untyped-def]
    """:func:`.solver.solve_u`: bisection over the whole range of u."""
    lo = cst[U_MIN]
    hi = cst[U_MAX]
    f_lo = _miss(lo, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out)
    f_hi = _miss(hi, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out)
    if not (f_hi <= 0.0 <= f_lo):
        return np.nan, NO_ROOT
    while hi - lo > cst[U_TOL]:
        mid = 0.5 * (lo + hi)
        if (
            _miss(
                mid, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
            )
            > 0.0
        ):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), OK


@njit(cache=True)
def _solve_u(target, guess, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out):  # type: ignore[no-untyped-def]
    """:func:`.solver.solve_u_near`: a bracket opened around the guess and
    widened until it holds the lap time, then Brent's method (scipy's
    ``brentq``, statement for statement)."""
    if not (math.isfinite(target) and math.isfinite(guess)):
        return _bisect_u(
            target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
        )
    u_min = cst[U_MIN]
    u_max = cst[U_MAX]
    lo = guess * 0.94
    if lo < u_min:
        lo = u_min
    hi = guess * 1.06
    if hi > u_max:
        hi = u_max
    f_lo = _miss(lo, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out)
    f_hi = _miss(hi, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out)
    for _ in range(8):
        if f_lo >= 0.0 >= f_hi:
            break
        if f_lo < 0.0:  # the lap is slower than the bracket's slow end
            if lo <= u_min:
                return np.nan, NO_ROOT
            hi = lo
            f_hi = f_lo
            lo = lo * 0.85
            if lo < u_min:
                lo = u_min
            f_lo = _miss(
                lo, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
            )
        else:
            if hi >= u_max:
                return np.nan, NO_ROOT
            lo = hi
            f_lo = f_hi
            hi = hi * 1.15
            if hi > u_max:
                hi = u_max
            f_hi = _miss(
                hi, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
            )
    if not (f_lo >= 0.0 >= f_hi):
        return _bisect_u(
            target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
        )
    if f_lo == 0.0:
        return lo, OK
    if f_hi == 0.0:
        return hi, OK
    xtol = cst[U_TOL]
    xpre = lo
    xcur = hi
    fpre = f_lo
    fcur = f_hi
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    for _ in range(BRENT_MAXITER):
        if fpre != 0.0 and fcur != 0.0 and ((fpre < 0.0) != (fcur < 0.0)):
            xblk = xpre
            fblk = fpre
            spre = xcur - xpre
            scur = spre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre
        delta = (xtol + BRENT_RTOL * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0.0 or abs(sbis) < delta:
            break
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:  # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:  # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            bound = 3.0 * abs(sbis) - delta
            if abs(spre) < bound:
                bound = abs(spre)
            if 2.0 * abs(stry) < bound:
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis
        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0.0 else -delta
        fcur = _miss(
            xcur, target, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
        )
    return xcur, OK


@njit(cache=True)
def solve_lane(target, guess, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out):  # type: ignore[no-untyped-def]
    """The u at which one lane laps in ``target`` and its status; the
    lap's node speeds at that u are left in ``out`` (NaN without a root)."""
    u, status = _solve_u(
        target, guess, kap, grd, kv, cos, gsb, ptab, mu_y, mu_b, w, c, kc, be, ons, cst, out
    )
    if status == OK:
        lap_nodes(
            kap, grd, kv, cos, gsb, ptab, mu_y * u * w, mu_b * u * w, c, kc, be, ons, cst, out
        )
    else:
        out[:] = np.nan
    return u, status


@njit(cache=True)
def solve_lanes(target, guess, kap, grd, kv, cos, gsb, ptabs, which, mu_y, mu_b, w, c, kc, be, ons, cst):  # type: ignore[no-untyped-def]
    """:func:`solve_lane` for every lane, one after the other."""
    lanes = len(target)
    u = np.empty(lanes)
    status = np.empty(lanes, np.int64)
    v = np.empty((lanes, len(kap)))
    for b in range(lanes):
        got = solve_lane(
            target[b], guess[b], kap, grd, kv, cos, gsb, ptabs[which[b]], mu_y[b], mu_b[b], w[b],
            c[b], kc[b], be[b], ons[b], cst, v[b],
        )  # fmt: skip
        u[b] = got[0]
        status[b] = got[1]
    return u, status, v


@njit(cache=True, parallel=True)
def solve_lanes_parallel(target, guess, kap, grd, kv, cos, gsb, ptabs, which, mu_y, mu_b, w, c, kc, be, ons, cst):  # type: ignore[no-untyped-def]
    """:func:`solve_lanes` with the lanes spread over the cores."""
    lanes = len(target)
    u = np.empty(lanes)
    status = np.empty(lanes, np.int64)
    v = np.empty((lanes, len(kap)))
    for b in prange(lanes):
        got = solve_lane(
            target[b], guess[b], kap, grd, kv, cos, gsb, ptabs[which[b]], mu_y[b], mu_b[b], w[b],
            c[b], kc[b], be[b], ons[b], cst, v[b],
        )  # fmt: skip
        u[b] = got[0]
        status[b] = got[1]
    return u, status, v


@njit(cache=True)
def drive_lanes(u, kap, grd, kv, cos, gsb, ptabs, which, mu_y, mu_b, w, c, kc, be, ons, cst):  # type: ignore[no-untyped-def]
    """Node speeds and lap time of every lane at its own ``u``."""
    lanes = len(u)
    t = np.empty(lanes)
    v = np.empty((lanes, len(kap)))
    for b in range(lanes):
        t[b] = lap_nodes(
            kap, grd, kv, cos, gsb, ptabs[which[b]], mu_y[b] * u[b] * w[b], mu_b[b] * u[b] * w[b],
            c[b], kc[b], be[b], ons[b], cst, v[b],
        )  # fmt: skip
    return t, v


@njit(cache=True, parallel=True)
def drive_lanes_parallel(u, kap, grd, kv, cos, gsb, ptabs, which, mu_y, mu_b, w, c, kc, be, ons, cst):  # type: ignore[no-untyped-def]
    """:func:`drive_lanes` with the lanes spread over the cores."""
    lanes = len(u)
    t = np.empty(lanes)
    v = np.empty((lanes, len(kap)))
    for b in prange(lanes):
        t[b] = lap_nodes(
            kap, grd, kv, cos, gsb, ptabs[which[b]], mu_y[b] * u[b] * w[b], mu_b[b] * u[b] * w[b],
            c[b], kc[b], be[b], ons[b], cst, v[b],
        )  # fmt: skip
    return t, v


__all__ = [
    "HAVE_NUMBA",
    "drive_lanes",
    "drive_lanes_parallel",
    "lap_nodes",
    "nearest_vertex",
    "open_peaks",
    "passes",
    "solve_lane",
    "sorted_bounds",
    "solve_lanes",
    "solve_lanes_parallel",
]
