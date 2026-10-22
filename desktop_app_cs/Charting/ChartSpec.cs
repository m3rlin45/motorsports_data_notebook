namespace InfernoAnalyzer.Charting;

/// <summary>
/// Which quantity the shared x-axis of a strip chart is plotted against.
/// </summary>
public enum XAxisMode
{
    /// <summary>Per-lap cumulative distance in metres (laps align spatially).</summary>
    Distance,

    /// <summary>Lap-relative time in seconds.</summary>
    Time,
}

/// <summary>
/// One line on a subplot. <see cref="X"/> and <see cref="Y"/> are the same
/// length and are slices of the caller's buffers; the spec never copies.
/// </summary>
/// <param name="Label">Legend label, e.g. "Lap 14".</param>
/// <param name="X">X values in data units (metres or seconds), non-decreasing.</param>
/// <param name="Y">Y values in the subplot's units.</param>
/// <param name="Color">CSS hex colour, e.g. "#2a78d6".</param>
public sealed record TraceSpec(
    string Label,
    ReadOnlyMemory<double> X,
    ReadOnlyMemory<double> Y,
    string Color
)
{
    /// <summary>
    /// The trace's value at <paramref name="x"/>, linearly interpolated
    /// between the two neighbouring samples. NaN outside the trace's x range
    /// (there is no data there, so a cursor label should not pretend).
    /// </summary>
    public double ValueAt(double x)
    {
        var xs = X.Span;
        var ys = Y.Span;
        if (xs.Length == 0 || x < xs[0] || x > xs[^1] || double.IsNaN(x))
        {
            return double.NaN;
        }

        // Index of the first sample at or after x.
        var hi = xs.BinarySearch(x);
        if (hi >= 0)
        {
            return ys[hi];
        }

        hi = ~hi; // 1 <= hi <= Length - 1 because x is strictly inside the range
        var lo = hi - 1;
        var span = xs[hi] - xs[lo];
        var t = span > 0 ? (x - xs[lo]) / span : 0;
        return ys[lo] + t * (ys[hi] - ys[lo]);
    }
}

/// <summary>
/// One stacked panel of a strip chart.
/// </summary>
/// <param name="Title">Panel title (channel name).</param>
/// <param name="Units">Y-axis unit label, e.g. "km/h".</param>
/// <param name="Traces">One trace per lap.</param>
/// <param name="ZeroLine">Draw a horizontal reference line at y = 0 (delta-time panels).</param>
/// <param name="Decimals">Decimal places for cursor value labels.</param>
public sealed record SubplotSpec(
    string Title,
    string Units,
    IReadOnlyList<TraceSpec> Traces,
    bool ZeroLine = false,
    int Decimals = 1
);

/// <summary>
/// Base type of the closed set of semantic chart specifications. Backends
/// switch on the concrete record type; adding a chart kind means adding a
/// record here and a case in every live backend.
/// </summary>
public abstract record ChartSpec;

/// <summary>
/// Vertically stacked subplots sharing one x-axis, with a linked crosshair
/// cursor. The defining view of a telemetry tool (MoTeC i2 / RaceStudio style).
/// </summary>
/// <param name="Subplots">Panels, top to bottom.</param>
/// <param name="XAxis">What the shared x-axis represents.</param>
public sealed record StripChartSpec(
    IReadOnlyList<SubplotSpec> Subplots,
    XAxisMode XAxis
) : ChartSpec
{
    /// <summary>Axis label for the bottom panel.</summary>
    public string XLabel => XAxis == XAxisMode.Distance ? "Distance (m)" : "Time (s)";

    /// <summary>Cursor position label, e.g. "2158 m" or "61.25 s".</summary>
    public string FormatX(double x) => XAxis == XAxisMode.Distance
        ? x.ToString("0", System.Globalization.CultureInfo.InvariantCulture) + " m"
        : x.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) + " s";
}

/// <summary>
/// A location reported by a surface, in data coordinates.
/// </summary>
/// <param name="X">X value in the chart's x units.</param>
/// <param name="Y">Y value in the subplot's units.</param>
/// <param name="SubplotIndex">Index of the panel the point is in.</param>
public readonly record struct ChartPoint(double X, double Y, int SubplotIndex);
