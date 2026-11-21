from __future__ import annotations

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


__all__ = [
    "HAVE_NUMBA",
    "nearest_vertex",
    "sorted_bounds",
]
