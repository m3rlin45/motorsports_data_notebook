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

/// <summary>An inclusive range on the x-axis, in data units.</summary>
public readonly record struct XRange(double Min, double Max)
{
    /// <summary>Whether <paramref name="x"/> lies inside the range.</summary>
    public bool Contains(double x) => x >= Min && x <= Max;
}

/// <summary>
/// One line on a subplot. <see cref="X"/> and <see cref="Y"/> are the same
/// length and are slices of the caller's buffers; the spec never copies.
/// </summary>
/// <param name="Label">Legend label, e.g. "Lap 14".</param>
/// <param name="X">X values in data units (metres or seconds), non-decreasing.</param>
/// <param name="Y">Y values in the subplot's units.</param>
/// <param name="Color">CSS hex colour, e.g. "#2a78d6".</param>
/// <param name="Emphasis">
/// The x range that is the trace's subject (the lap itself); samples outside
/// it are context (neighbouring laps) and are drawn faded. Null: all of it.
/// </param>
public sealed record TraceSpec(
    string Label,
    ReadOnlyMemory<double> X,
    ReadOnlyMemory<double> Y,
    string Color,
    XRange? Emphasis = null
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
/// A labelled vertical delimiter drawn on every panel, e.g. a lap boundary.
/// </summary>
/// <param name="X">Position in data units.</param>
/// <param name="Label">Short label, e.g. "Lap 13".</param>
/// <param name="Color">CSS hex colour.</param>
/// <param name="Row">
/// Label row (0 = nearest the axis). Markers from different laps share x
/// positions (every lap starts at 0), so each lap's labels get their own row.
/// </param>
public sealed record XMarkerSpec(double X, string Label, string Color, int Row = 0);

/// <summary>
/// Vertically stacked subplots sharing one x-axis, with a linked crosshair
/// cursor. The defining view of a telemetry tool (MoTeC i2 / RaceStudio style).
/// </summary>
/// <param name="Subplots">Panels, top to bottom.</param>
/// <param name="XAxis">What the shared x-axis represents.</param>
/// <param name="Markers">Vertical delimiters shared by all panels (lap boundaries).</param>
/// <param name="InitialView">
/// X range shown after rendering; the user pans/zooms from there. Null: fit
/// all data.
/// </param>
/// <param name="Legend">
/// Show the backend's legend on the top panel. Off when the view provides
/// its own key (the Time/Distance lap picker), since an in-plot legend
/// covers the data once many laps are overlaid.
/// </param>
public sealed record StripChartSpec(
    IReadOnlyList<SubplotSpec> Subplots,
    XAxisMode XAxis,
    IReadOnlyList<XMarkerSpec>? Markers = null,
    XRange? InitialView = null,
    bool Legend = true
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

/// <summary>A shaded vertical band between two x values (e.g. a velocity range).</summary>
public sealed record BandSpec(double Start, double End, string Color, double Alpha = 0.2, string? Label = null);

/// <summary>One bar series: bars centred on <see cref="Centers"/> with heights <see cref="Values"/>.</summary>
/// <param name="Label">Legend label (empty: no legend entry).</param>
/// <param name="Centers">Bar centres in x units.</param>
/// <param name="Values">Bar heights in y units.</param>
/// <param name="Color">CSS hex colour for every bar, or null to use <see cref="BarColors"/>.</param>
/// <param name="BarColors">Per-bar colours when the series is not single-coloured.</param>
public sealed record BarSeriesSpec(
    string Label,
    double[] Centers,
    double[] Values,
    string? Color = null,
    string[]? BarColors = null);

/// <summary>One panel of a histogram grid.</summary>
/// <param name="Title">Panel title, e.g. "Front Left (FL)".</param>
/// <param name="Series">Bar series; several are drawn side by side within each bin.</param>
/// <param name="BinWidth">Bin width in x units (bars are drawn narrower).</param>
/// <param name="Annotation">Text in the panel's top-right corner, or empty.</param>
public sealed record HistogramPanelSpec(string Title, IReadOnlyList<BarSeriesSpec> Series, double BinWidth, string Annotation = "");

/// <summary>
/// A grid of bar histograms sharing x limits and background bands; the
/// suspension velocity view (2×2 corners, one or two sessions per panel).
/// </summary>
/// <param name="Title">Figure title.</param>
/// <param name="Panels">Panels in row-major order.</param>
/// <param name="Columns">Panels per row.</param>
/// <param name="XLimits">Shared x range.</param>
/// <param name="XLabel">X axis label (bottom row).</param>
/// <param name="YLabel">Y axis label (left column).</param>
/// <param name="Bands">Shaded ranges drawn behind every panel.</param>
public sealed record HistogramGridSpec(
    string Title,
    IReadOnlyList<HistogramPanelSpec> Panels,
    int Columns,
    XRange XLimits,
    string XLabel,
    string YLabel,
    IReadOnlyList<BandSpec> Bands) : ChartSpec;

/// <summary>A cloud of points; samples with the same <see cref="SeriesId"/> are index-aligned across panels and the map.</summary>
public sealed record ScatterSeriesSpec(string Label, double[] X, double[] Y, string Color, int SeriesId, double Alpha = 0.2);

/// <summary>A straight line between two points (a fit line).</summary>
public sealed record LineSpec(double X1, double Y1, double X2, double Y2, string Color);

/// <summary>Emphasised markers (binned means).</summary>
public sealed record MarkerSeriesSpec(double[] X, double[] Y, string Color);

/// <summary>One scatter panel.</summary>
public sealed record ScatterPanelSpec(
    string Title,
    string XLabel,
    string YLabel,
    IReadOnlyList<ScatterSeriesSpec> Scatter,
    IReadOnlyList<LineSpec> Lines,
    IReadOnlyList<MarkerSeriesSpec> Markers,
    string Annotation = "",
    string Warning = "",
    bool ShowLegend = false);

/// <summary>A track outline in local metres, index-aligned with the scatter series of the same <see cref="SeriesId"/>.</summary>
public sealed record TrackPathSpec(string Label, double[] X, double[] Y, string Color, string HighlightColor, int SeriesId);

/// <summary>
/// A track map plus scatter panels in one grid. Hovering the map near the
/// track highlights the samples recorded there on every panel (the
/// roll-gradient view).
/// </summary>
/// <param name="Title">Figure title.</param>
/// <param name="Paths">Track paths for the map (empty: the map says there is no GPS).</param>
/// <param name="Panels">Scatter panels after the map, row-major.</param>
/// <param name="Columns">Panels per row (the map is the first cell).</param>
public sealed record ScatterGridSpec(
    string Title,
    IReadOnlyList<TrackPathSpec> Paths,
    IReadOnlyList<ScatterPanelSpec> Panels,
    int Columns) : ChartSpec;

/// <summary>One box of a box-and-whisker plot, with its statistics precomputed (numpy percentile semantics).</summary>
public sealed record BoxSpec(
    double Position,
    double Q1,
    double Median,
    double Q3,
    double WhiskerLow,
    double WhiskerHigh,
    double Mean,
    double[] Fliers,
    string Color,
    string Tooltip,
    double Width = 0.5)
{
    /// <summary>Box statistics of <paramref name="values"/>: quartiles by linear interpolation, whiskers at 1.5 IQR, the rest fliers.</summary>
    public static BoxSpec From(double position, IReadOnlyList<double> values, string color, string tooltip, double width = 0.5)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        double Pct(double q)
        {
            var index = q / 100.0 * (sorted.Length - 1);
            var lo = (int)Math.Floor(index);
            var hi = Math.Min(lo + 1, sorted.Length - 1);
            var t = index - lo;
            return t >= 0.5 ? sorted[hi] - (sorted[hi] - sorted[lo]) * (1 - t) : sorted[lo] + (sorted[hi] - sorted[lo]) * t;
        }

        var q1 = Pct(25);
        var median = Pct(50);
        var q3 = Pct(75);
        var iqr = q3 - q1;
        var low = sorted.Where(v => v >= q1 - 1.5 * iqr).DefaultIfEmpty(q1).Min();
        var high = sorted.Where(v => v <= q3 + 1.5 * iqr).DefaultIfEmpty(q3).Max();
        var fliers = sorted.Where(v => v < low || v > high).ToArray();
        var mean = sorted.Length > 0 ? sorted.Average() : double.NaN;
        return new BoxSpec(position, q1, median, q3, low, high, mean, fliers, color, tooltip, width);
    }
}

/// <summary>One stacked panel of boxes.</summary>
public sealed record BoxPanelSpec(string Title, string YLabel, IReadOnlyList<BoxSpec> Boxes, bool ZeroLine = false);

/// <summary>Stacked box-plot panels sharing the category axis (the driver summary view).</summary>
/// <param name="Title">Figure title.</param>
/// <param name="Panels">Panels top to bottom.</param>
/// <param name="CategoryPositions">X position of each category.</param>
/// <param name="CategoryLabels">Label of each category (bottom panel ticks).</param>
/// <param name="Bands">Shaded category bands (top-opportunity corners), drawn on every panel.</param>
/// <param name="Legend">Legend entries (empty: no legend).</param>
public sealed record BoxPlotSpec(
    string Title,
    IReadOnlyList<BoxPanelSpec> Panels,
    double[] CategoryPositions,
    string[] CategoryLabels,
    IReadOnlyList<BandSpec> Bands,
    IReadOnlyList<(string Label, string Color)> Legend) : ChartSpec;

/// <summary>A coloured polyline on a track map.</summary>
public sealed record TrackMapPathSpec(double[] X, double[] Y, string Color, double Width, string Label = "");

/// <summary>A marked point on a track map.</summary>
public sealed record TrackMapPointSpec(double X, double Y, string Color, double Size, string Label = "", bool Star = false);

/// <summary>A standalone track map with coloured segments and marked points (equal units).</summary>
public sealed record TrackMapSpec(string Title, IReadOnlyList<TrackMapPathSpec> Paths, IReadOnlyList<TrackMapPointSpec> Points) : ChartSpec;

/// <summary>A line with markers through (X, Y), each point carrying a sample count for the hover label.</summary>
public sealed record LineSeriesSpec(string Label, double[] X, double[] Y, long[] Counts, string Color);

/// <summary>One panel of a line grid.</summary>
public sealed record LinePanelSpec(string Title, IReadOnlyList<LineSeriesSpec> Series, string Annotation = "");

/// <summary>A grid of line panels sharing axis labels (the tire-grip view).</summary>
public sealed record LineGridSpec(string Title, IReadOnlyList<LinePanelSpec> Panels, int Columns, string XLabel, string YLabel) : ChartSpec;

