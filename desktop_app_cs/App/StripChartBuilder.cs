using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App;

/// <summary>
/// Turns a <see cref="Fixture"/> into a <see cref="StripChartSpec"/>: one panel
/// per channel with every lap overlaid, plus a delta-time panel against the
/// reference (first) lap. Pure data in, pure data out — no UI types — so it is
/// unit-testable without a display.
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
        var subplots = new List<SubplotSpec>();

        foreach (var channel in fixture.Channels)
        {
            var traces = new List<TraceSpec>();
            for (var i = 0; i < fixture.Laps.Count; i++)
            {
                var lap = fixture.Laps[i];
                if (!lap.Series.TryGetValue(channel, out var y))
                {
                    continue;
                }

                var x = xAxis == XAxisMode.Distance ? lap.DistanceM : lap.TimeS;
                traces.Add(new TraceSpec(lap.Label, x, y, ColorFor(i)));
            }

            fixture.Units.TryGetValue(channel, out var units);
            subplots.Add(new SubplotSpec(channel, units ?? string.Empty, traces));
        }

        if (fixture.Laps.Count >= 2)
        {
            subplots.Add(BuildDeltaTime(fixture, xAxis));
        }

        return new StripChartSpec(subplots, xAxis);
    }

    /// <summary>Palette colour for lap index <paramref name="i"/>.</summary>
    public static string ColorFor(int i) => Palette[i % Palette.Length];

    /// <summary>
    /// Delta-time panel: for each non-reference lap, time gained (negative) or
    /// lost (positive) relative to the reference lap as a function of
    /// distance. Both laps are resampled onto a common distance grid; the
    /// delta at distance d is t_lap(d) − t_ref(d). In time mode the x-axis is
    /// the reference lap's elapsed time at that distance, so the panel still
    /// lines up with the reference lap's traces.
    /// </summary>
    public static SubplotSpec BuildDeltaTime(Fixture fixture, XAxisMode xAxis)
    {
        var reference = fixture.Laps[0];
        var maxDistance = fixture.Laps.Min(lap => lap.DistanceM[^1]);
        var grid = Grid(0, maxDistance, DeltaGridStepM);
        var referenceTime = Interpolate(reference.DistanceM, reference.TimeS, grid);
        var x = xAxis == XAxisMode.Distance ? grid : referenceTime;

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

            traces.Add(new TraceSpec($"{lap.Label} vs Lap {reference.Num}", x, delta, ColorFor(i)));
        }

        return new SubplotSpec("Delta time vs reference", "s", traces, ZeroLine: true);
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
