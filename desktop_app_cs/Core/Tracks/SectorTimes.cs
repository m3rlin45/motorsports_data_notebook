using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Tracks;

/// <summary>One lap's sector times.</summary>
/// <param name="LapNum">Lap number.</param>
/// <param name="LapTimeS">Lap time from the laps table.</param>
/// <param name="Sectors">Sector times in seconds; NaN where a gate was not crossed.</param>
/// <param name="GateTimesS">Lap-relative crossing time of each gate, NaN when not crossed.</param>
public sealed record LapSplits(int LapNum, double LapTimeS, double[] Sectors, double[] GateTimesS)
{
    /// <summary>Every gate was crossed in order.</summary>
    public bool Complete => Sectors.All(s => !double.IsNaN(s));
}

/// <summary>The split-times table for a set of laps on a track.</summary>
/// <param name="Track">The track.</param>
/// <param name="Laps">Rows, in the order requested.</param>
/// <param name="BestSector">Fastest time per sector over the complete laps (NaN when none).</param>
/// <param name="TheoreticalBestS">Sum of the best sectors (NaN when any sector has none).</param>
/// <param name="BestLapNum">Lap number of the fastest complete lap, or null.</param>
public sealed record SplitTable(TrackDefinition Track, IReadOnlyList<LapSplits> Laps, double[] BestSector, double TheoreticalBestS, int? BestLapNum);

/// <summary>Sector times from GPS paths crossing gate chords.</summary>
public static class SectorTimes
{
    /// <summary>
    /// Crossing times of the track's gates along a lap's GPS path, in gate
    /// order (each gate is searched after the previous crossing). The path is
    /// lap-filtered GPS samples; times are ms on the session clock.
    /// </summary>
    public static double[] GateCrossingsS(ReadOnlySpan<double> lat, ReadOnlySpan<double> lon, ReadOnlySpan<long> timecodes, TrackDefinition track, long lapStartMs)
    {
        var origin = track.Anchor;
        var n = lat.Length;
        var x = new double[n];
        var y = new double[n];
        for (var i = 0; i < n; i++)
        {
            (x[i], y[i]) = GeoProjection.ToLocal(new GeoPoint(lat[i], lon[i]), origin);
        }

        var result = new double[track.Gates.Count];
        Array.Fill(result, double.NaN);
        var from = 0;
        for (var g = 0; g < track.Gates.Count; g++)
        {
            var gate = track.Gates[g];
            var (ax, ay) = GeoProjection.ToLocal(gate.A, origin);
            var (bx, by) = GeoProjection.ToLocal(gate.B, origin);
            for (var i = from; i < n - 1; i++)
            {
                if (SegmentParameter(x[i], y[i], x[i + 1], y[i + 1], ax, ay, bx, by) is { } t)
                {
                    var crossMs = timecodes[i] + t * (timecodes[i + 1] - timecodes[i]);
                    result[g] = (crossMs - lapStartMs) / 1000.0;
                    from = i + 1;
                    break;
                }
            }

            if (double.IsNaN(result[g]))
            {
                break; // later gates cannot be ordered after a missing one
            }
        }

        return result;
    }

    /// <summary>Parameter t in [0, 1] along P→Q where it crosses segment A→B, or null when they do not intersect.</summary>
    public static double? SegmentParameter(double px, double py, double qx, double qy, double ax, double ay, double bx, double by)
    {
        var rx = qx - px;
        var ry = qy - py;
        var sx = bx - ax;
        var sy = by - ay;
        var denominator = rx * sy - ry * sx;
        if (Math.Abs(denominator) < 1e-12)
        {
            return null; // parallel
        }

        var t = ((ax - px) * sy - (ay - py) * sx) / denominator;
        var u = ((ax - px) * ry - (ay - py) * rx) / denominator;
        return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? t : null;
    }

    /// <summary>Sector times for one lap (sector i = gate i crossing − previous crossing; the last to the lap end).</summary>
    public static LapSplits ForLap(Lap lap, ReadOnlySpan<double> lat, ReadOnlySpan<double> lon, ReadOnlySpan<long> timecodes, TrackDefinition track)
    {
        var gates = GateCrossingsS(lat, lon, timecodes, track, lap.StartTime);
        var sectors = new double[track.SectorCount];
        var previous = 0.0;
        for (var g = 0; g < gates.Length; g++)
        {
            sectors[g] = double.IsNaN(gates[g]) ? double.NaN : gates[g] - previous;
            previous = gates[g];
        }

        sectors[^1] = gates.Length > 0 && double.IsNaN(gates[^1]) ? double.NaN : lap.LapTimeS - previous;
        return new LapSplits(lap.Num, lap.LapTimeS, sectors, gates);
    }

    /// <summary>The split table for <paramref name="lapNums"/> of a session; <paramref name="cancellation"/> is checked between laps.</summary>
    public static SplitTable Build(LogFile log, IEnumerable<int> lapNums, TrackDefinition track, string latChannel, string lonChannel, CancellationToken cancellation = default)
    {
        var rows = new List<LapSplits>();
        foreach (var num in lapNums)
        {
            cancellation.ThrowIfCancellationRequested();
            var lap = log.Laps.FirstOrDefault(l => l.Num == num);
            if (lap is null)
            {
                continue;
            }

            var filtered = log.FilterByLap(num).SelectChannels(new[] { latChannel, lonChannel }).ResampleToChannel(latChannel);
            rows.Add(ForLap(lap, filtered[latChannel].Values, filtered[lonChannel].Values, filtered[latChannel].Timecodes, track));
        }

        var best = new double[track.SectorCount];
        for (var s = 0; s < best.Length; s++)
        {
            var times = rows.Where(r => r.Complete).Select(r => r.Sectors[s]).ToList();
            best[s] = times.Count > 0 ? times.Min() : double.NaN;
        }

        var theoretical = best.Any(double.IsNaN) ? double.NaN : best.Sum();
        var bestLap = rows.Where(r => r.Complete && r.LapTimeS > 0).OrderBy(r => r.LapTimeS).Select(r => (int?)r.LapNum).FirstOrDefault();
        return new SplitTable(track, rows, best, theoretical, bestLap);
    }

    /// <summary>
    /// A gate across the path at <paramref name="index"/>: a chord of
    /// <paramref name="halfWidthM"/> metres either side, perpendicular to the
    /// direction of travel (the track editor's "add gate here").
    /// </summary>
    public static SectorGate GateAcrossPath(ReadOnlySpan<double> lat, ReadOnlySpan<double> lon, int index, string name, GeoPoint origin, double halfWidthM = 25)
    {
        var i0 = Math.Max(index - 1, 0);
        var i1 = Math.Min(index + 1, lat.Length - 1);
        var (x0, y0) = GeoProjection.ToLocal(new GeoPoint(lat[i0], lon[i0]), origin);
        var (x1, y1) = GeoProjection.ToLocal(new GeoPoint(lat[i1], lon[i1]), origin);
        var (cx, cy) = GeoProjection.ToLocal(new GeoPoint(lat[index], lon[index]), origin);
        var dx = x1 - x0;
        var dy = y1 - y0;
        var len = Math.Sqrt(dx * dx + dy * dy);
        if (len < 1e-9)
        {
            dx = 1;
            dy = 0;
            len = 1;
        }

        var nx = -dy / len;
        var ny = dx / len;
        var a = GeoProjection.FromLocal(cx + nx * halfWidthM, cy + ny * halfWidthM, origin);
        var b = GeoProjection.FromLocal(cx - nx * halfWidthM, cy - ny * halfWidthM, origin);
        return new SectorGate(name, a, b);
    }
}
