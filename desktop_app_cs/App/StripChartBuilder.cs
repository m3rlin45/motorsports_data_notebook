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
    /// <summary>Grid spacing of the distance resample used for delta-time, metres.</summary>
    public const double DeltaGridStepM = 5.0;

    /// <summary>Panels with more laps than this drop the per-lap boundary labels: they would overprint each other.</summary>
    public const int MaxLabelledLaps = 2;

    /// <summary>
    /// Build the spec for <paramref name="fixture"/> on the given x-axis.
    /// <paramref name="viewKey"/> names what is on show (the session file),
    /// so re-building for the same thing keeps the user's zoom.
    /// </summary>
    public static StripChartSpec Build(Fixture fixture, XAxisMode xAxis, string? viewKey = null)
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
                var style = StyleFor(fixture, i);
                traces.Add(new TraceSpec(lap.Label, s.X, s.Series(channel), style.Color, s.LapRange, style.Dashed, Reference: i == 0, ShortLabel: lap.Tag));
            }

            fixture.Units.TryGetValue(channel, out var units);
            subplots.Add(new SubplotSpec(DisplayName(channel), string.IsNullOrEmpty(units) ? DefaultUnits(channel) : units, traces));
        }

        // Delta time is "time lost at this point on track", which only means
        // something against distance: against time the laps are not aligned.
        if (xAxis == XAxisMode.Distance && fixture.Laps.Count >= 2)
        {
            subplots.Add(BuildDeltaTime(fixture));
        }

        // Lap boundaries. With one or two laps each gets its own labelled lines (which lap is before
        // and after); with more, the labels would stack into a smear, so one neutral start/finish
        // line and the reference lap's end stand for all of them.
        var markers = new List<XMarkerSpec>();
        if (stitched.Count <= MaxLabelledLaps)
        {
            for (var i = 0; i < stitched.Count; i++)
            {
                markers.AddRange(stitched[i].Markers(StyleFor(fixture, i).Color, row: i));
            }
        }
        else
        {
            markers.Add(new XMarkerSpec(0, "Start / finish", ChartColors.Neutral));
            markers.Add(new XMarkerSpec(stitched[0].LapRange.Max, string.Empty, ChartColors.Neutral));
        }

        var view = stitched.Count == 0
            ? (XRange?)null
            : new XRange(0, stitched.Max(s => s.LapRange.Max));

        return new StripChartSpec(subplots, xAxis, markers, view, ViewKey: viewKey);
    }

    /// <summary>
    /// The style of lap <paramref name="i"/>: the one its session's lap list
    /// shows, or its position's when the fixture carries none (a dumped file).
    /// </summary>
    public static LapStyle StyleFor(Fixture fixture, int i) =>
        fixture.Laps[i].Style ?? new LapStyle(i % LapStyleAllocator.SlotCount);

    /// <summary>Units of the derived channels, which carry none of their own.</summary>
    public static string DefaultUnits(string channel) => channel switch
    {
        "speed_kmh" => "km/h",
        "distance_m" => "m",
        _ => string.Empty,
    };

    /// <summary>Panel name of a channel: the derived channels get a readable one, logged channels keep the logger's name.</summary>
    public static string DisplayName(string channel) => channel switch
    {
        "speed_kmh" => "Speed",
        "distance_m" => "Distance",
        _ => channel,
    };

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

            var style = StyleFor(fixture, i);
            traces.Add(new TraceSpec($"{lap.Label} vs Lap {reference.Num}", grid, delta, style.Color, Dashed: style.Dashed, ShortLabel: lap.Tag));
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
