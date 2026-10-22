using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App;

/// <summary>
/// Turns a <see cref="Fixture"/> into a <see cref="StripChartSpec"/>: one panel
/// per channel with every lap overlaid, plus (in distance mode) a delta-time
/// panel against the reference (first) lap. Pure data in, pure data out — no
/// UI types — so it is unit-testable without a display.
///
/// Each lap's trace carries its neighbouring laps on either side so the view
/// can scroll past the lap boundaries: the preceding lap sits at negative x,
/// the following lap after the lap end, and lap boundaries become markers.
/// </summary>
public static class StripChartBuilder
{
    /// <summary>Validated categorical palette (dataviz skill), one colour per lap.</summary>
    public static readonly string[] Palette =
    {
        "#2a78d6", "#eb6834", "#2ca25f", "#b56bd6", "#d6a52a", "#38b6c9", "#d64a6e", "#8a8a8a",
    };

    /// <summary>Grid spacing of the distance resample used for delta-time, metres.</summary>
    public const double DeltaGridStepM = 5.0;

    /// <summary>Build the spec for <paramref name="fixture"/> on the given x-axis.</summary>
    public static StripChartSpec Build(Fixture fixture, XAxisMode xAxis)
    {
        var stitched = fixture.Laps.Select(lap => Stitch(lap, xAxis)).ToList();
        var subplots = new List<SubplotSpec>();

        foreach (var channel in fixture.Channels)
        {
            var traces = new List<TraceSpec>();
            for (var i = 0; i < fixture.Laps.Count; i++)
            {
                var lap = fixture.Laps[i];
                if (!lap.Series.ContainsKey(channel))
                {
                    continue;
                }

                var s = stitched[i];
                traces.Add(new TraceSpec(lap.Label, s.X, s.Series(channel), ColorFor(i), s.LapRange));
            }

            fixture.Units.TryGetValue(channel, out var units);
            subplots.Add(new SubplotSpec(channel, units ?? string.Empty, traces));
        }

        // Delta time is "time lost at this point on track", which only means
        // something against distance: against time the laps are not aligned.
        if (xAxis == XAxisMode.Distance && fixture.Laps.Count >= 2)
        {
            subplots.Add(BuildDeltaTime(fixture));
        }

        var markers = new List<XMarkerSpec>();
        for (var i = 0; i < stitched.Count; i++)
        {
            markers.AddRange(stitched[i].Markers(ColorFor(i), row: i));
        }

        var view = stitched.Count == 0
            ? (XRange?)null
            : new XRange(0, stitched.Max(s => s.LapRange.Max));

        return new StripChartSpec(subplots, xAxis, markers, view);
    }

    /// <summary>Palette colour for lap index <paramref name="i"/>.</summary>
    public static string ColorFor(int i) => Palette[i % Palette.Length];

    /// <summary>
    /// A lap with its neighbours concatenated into one x-sorted series:
    /// [before (x shifted by −before length)] [lap] [after (x shifted by +lap length)].
    /// </summary>
    public sealed class StitchedLap
    {
        internal StitchedLap(FixtureLap lap, XAxisMode xAxis)
        {
            Lap = lap;
            var segments = new List<(FixtureLap Lap, double Offset)>();
            if (lap.Before is { } before)
            {
                segments.Add((before, -Length(before, xAxis)));
            }

            segments.Add((lap, 0));
            if (lap.After is { } after)
            {
                segments.Add((after, Length(lap, xAxis)));
            }

            _segments = segments;
            X = Concat(segments, (l, offset) => Axis(l, xAxis).Select(v => v + offset));
            LapRange = new XRange(0, Length(lap, xAxis));
        }

        private readonly List<(FixtureLap Lap, double Offset)> _segments;

        /// <summary>The lap itself.</summary>
        public FixtureLap Lap { get; }

        /// <summary>X of every sample, neighbours included.</summary>
        public double[] X { get; }

        /// <summary>The x range occupied by the lap itself.</summary>
        public XRange LapRange { get; }

        /// <summary>Samples of <paramref name="channel"/> aligned with <see cref="X"/>.</summary>
        public double[] Series(string channel) =>
            Concat(_segments, (l, _) => l.Series.TryGetValue(channel, out var ys) ? ys : Enumerable.Repeat(double.NaN, l.TimeS.Length));

        /// <summary>One marker per lap boundary, labelled with the lap that starts there.</summary>
        public IEnumerable<XMarkerSpec> Markers(string color, int row)
        {
            foreach (var (lap, offset) in _segments)
            {
                yield return new XMarkerSpec(offset, $"Lap {lap.Num}", color, row);
            }

            if (Lap.After is null)
            {
                yield return new XMarkerSpec(LapRange.Max, $"Lap {Lap.Num} end", color, row);
            }
        }

        private static double[] Axis(FixtureLap lap, XAxisMode xAxis) =>
            xAxis == XAxisMode.Distance ? lap.DistanceM : lap.TimeS;

        // Distance: the last sample's distance (the lap's length as logged).
        // Time: the lap time, which is what the next lap's clock restarts from.
        private static double Length(FixtureLap lap, XAxisMode xAxis) =>
            xAxis == XAxisMode.Distance ? lap.DistanceM[^1] : lap.LapTimeS;

        private static double[] Concat(
            List<(FixtureLap Lap, double Offset)> segments,
            Func<FixtureLap, double, IEnumerable<double>> select) =>
            segments.SelectMany(s => select(s.Lap, s.Offset)).ToArray();
    }

    /// <summary>Stitch <paramref name="lap"/> to its neighbours on the given axis.</summary>
    public static StitchedLap Stitch(FixtureLap lap, XAxisMode xAxis) => new(lap, xAxis);

    /// <summary>
    /// Delta-time panel: for each non-reference lap, time gained (negative) or
    /// lost (positive) relative to the reference lap as a function of
    /// distance. Both laps are resampled onto a common distance grid; the
    /// delta at distance d is t_lap(d) − t_ref(d).
    /// </summary>
    public static SubplotSpec BuildDeltaTime(Fixture fixture)
    {
        var reference = fixture.Laps[0];
        var maxDistance = fixture.Laps.Min(lap => lap.DistanceM[^1]);
        var grid = Grid(0, maxDistance, DeltaGridStepM);
        var referenceTime = Interpolate(reference.DistanceM, reference.TimeS, grid);

        var traces = new List<TraceSpec>();
        for (var i = 1; i < fixture.Laps.Count; i++)
        {
            var lap = fixture.Laps[i];
            var lapTime = Interpolate(lap.DistanceM, lap.TimeS, grid);
            var delta = new double[grid.Length];
            for (var k = 0; k < grid.Length; k++)
            {
                delta[k] = lapTime[k] - referenceTime[k];
            }

            traces.Add(new TraceSpec($"{lap.Label} vs Lap {reference.Num}", grid, delta, ColorFor(i)));
        }

        return new SubplotSpec("Delta time vs reference", "s", traces, ZeroLine: true, Decimals: 2);
    }

    /// <summary>Uniform grid from <paramref name="start"/> to <paramref name="end"/> inclusive.</summary>
    public static double[] Grid(double start, double end, double step)
    {
        var count = (int)Math.Floor((end - start) / step) + 1;
        var grid = new double[Math.Max(count, 1)];
        for (var i = 0; i < grid.Length; i++)
        {
            grid[i] = start + i * step;
        }

        return grid;
    }

    /// <summary>
    /// Linear interpolation of (<paramref name="xs"/>, <paramref name="ys"/>) at
    /// <paramref name="at"/>, numpy.interp semantics: xs must be non-decreasing;
    /// values outside the range clamp to the end values.
    /// </summary>
    public static double[] Interpolate(ReadOnlySpan<double> xs, ReadOnlySpan<double> ys, ReadOnlySpan<double> at)
    {
        var result = new double[at.Length];
        var j = 0;
        for (var i = 0; i < at.Length; i++)
        {
            var x = at[i];
            if (x <= xs[0])
            {
                result[i] = ys[0];
                continue;
            }

            if (x >= xs[^1])
            {
                result[i] = ys[^1];
                continue;
            }

            while (j < xs.Length - 2 && xs[j + 1] < x)
            {
                j++;
            }

            var span = xs[j + 1] - xs[j];
            var t = span > 0 ? (x - xs[j]) / span : 0;
            result[i] = ys[j] + t * (ys[j + 1] - ys[j]);
        }

        return result;
    }
}
