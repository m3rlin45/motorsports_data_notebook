using InfernoAnalyzer.Charting;
using ScottPlot;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// How every track map is drawn, whichever chart it is part of: no axes or
/// grid (the coordinates are metres from an arbitrary origin and mean
/// nothing to the reader), equal metres per pixel, a scale bar, and the
/// title as a caption inside the frame.
/// </summary>
internal static class MapStyle
{
    /// <summary>Width of the neutral track outline.</summary>
    public const float OutlineWidth = 2;

    /// <summary>Clear <paramref name="plot"/> and style it as a map.</summary>
    public static void Begin(Plot plot, ChartTheme theme, string caption)
    {
        plot.Clear();
        PlotStyle.Apply(plot, theme);
        plot.Axes.Frameless();
        plot.HideGrid();
        plot.Axes.Rules.Clear();
        if (caption.Length > 0)
        {
            PlotStyle.Caption(plot, theme, caption, Alignment.UpperLeft, theme.Axis);
        }
    }

    /// <summary>
    /// Fit the data with equal units and add a scale bar sized to a round
    /// length near a fifth of the map's extent.
    /// </summary>
    public static void Finish(Plot plot, ChartTheme theme)
    {
        plot.Axes.AutoScale();
        var limits = plot.Axes.GetLimits();
        var extent = Math.Max(limits.HorizontalSpan, limits.VerticalSpan);
        if (double.IsFinite(extent) && extent > 0)
        {
            var length = RoundLength(extent / 5);
            var bar = plot.Add.ScaleBar(length, 0);
            bar.XLabel = length >= 1000 ? $"{length / 1000:0.#} km" : $"{length:0} m";
            bar.YLabel = string.Empty;
            bar.LineColor = PlotStyle.Color(theme, theme.Axis);
            bar.LineWidth = 1.5f;
            bar.XLabelStyle.ForeColor = PlotStyle.Color(theme, theme.Axis);
            bar.XLabelStyle.FontSize = theme.TickSize;
        }

        // Equal metres per pixel; zoom the shorter axis out rather than cropping the longer one.
        plot.Axes.Rules.Add(new global::ScottPlot.AxisRules.SquareZoomOut(plot.Axes.Bottom, plot.Axes.Left));
    }

    /// <summary>The largest of 1, 2 or 5 × 10ⁿ not above <paramref name="metres"/>.</summary>
    internal static double RoundLength(double metres)
    {
        var magnitude = Math.Pow(10, Math.Floor(Math.Log10(metres)));
        var fraction = metres / magnitude;
        return (fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1) * magnitude;
    }
}
