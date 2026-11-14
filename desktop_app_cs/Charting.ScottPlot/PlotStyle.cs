using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// The one chart style every ScottPlot surface shares: surfaces and ink from
/// the <see cref="ChartTheme"/>, a quiet axis (left and bottom frame in the
/// grid colour, no minor ticks, regular-weight captions smaller than the
/// data), flat legend and label boxes, and the UI font.
/// </summary>
internal static class PlotStyle
{
    /// <summary>Corner radius of label and annotation boxes.</summary>
    public const float BoxRadius = 4;

    /// <summary>Resolve a spec colour (hex or <see cref="ChartColors"/> token) for <paramref name="theme"/>.</summary>
    public static Color Color(ChartTheme theme, string color) => global::ScottPlot.Color.FromHex(theme.Resolve(color));

    /// <summary>Opacity of translucent fills: opaque in high-contrast themes.</summary>
    public static double Alpha(ChartTheme theme, double alpha) => theme.HighContrast ? 1 : alpha;

    /// <summary>Style <paramref name="plot"/>'s surfaces, axes, grid, title and legend.</summary>
    public static void Apply(Plot plot, ChartTheme theme)
    {
        ScottPlotFonts.UseFamily(theme.FontFamily);

        var axis = Color(theme, theme.Axis);
        var grid = Color(theme, theme.Grid);
        var text = Color(theme, theme.Text);

        plot.FigureBackground.Color = Color(theme, theme.FigureBackground);
        plot.DataBackground.Color = Color(theme, theme.DataBackground);
        plot.Axes.Color(axis);
        foreach (var a in plot.Axes.GetAxes())
        {
            a.Label.FontSize = theme.LabelSize;
            a.Label.Bold = false;
            a.Label.ForeColor = axis;
            a.TickLabelStyle.FontSize = theme.TickSize;
            a.TickLabelStyle.ForeColor = axis;
            a.MinorTickStyle.Length = 0;
            a.MajorTickStyle.Length = 3;
            a.MajorTickStyle.Color = theme.HighContrast ? axis : grid;
            a.FrameLineStyle.Color = theme.HighContrast ? axis : grid;
            a.FrameLineStyle.Width = 1;
        }

        // Only the two edges that carry a scale are drawn.
        plot.Axes.Top.FrameLineStyle.Width = 0;
        plot.Axes.Right.FrameLineStyle.Width = 0;

        plot.Axes.Title.Label.ForeColor = text;
        plot.Axes.Title.Label.FontSize = theme.TitleSize;
        plot.Axes.Title.Label.Bold = true;
        plot.Grid.MajorLineColor = grid;

        plot.Legend.BackgroundColor = Color(theme, theme.PanelBackground).WithAlpha(Alpha(theme, 0.9));
        plot.Legend.OutlineColor = Color(theme, theme.PanelOutline);
        plot.Legend.FontColor = text;
        plot.Legend.FontSize = theme.TickSize;
        plot.Legend.ShadowColor = Colors.Transparent;
    }

    /// <summary>
    /// Pick a font per label from the characters it holds, so a session
    /// named in Japanese still draws (Skia does no per-glyph fallback). Call
    /// once every label of the plot has its text.
    /// </summary>
    public static void FitFonts(Plot plot) => plot.Font.Automatic();

    /// <summary>A hidden text box that follows the pointer; the caller sets its text and location on hover.</summary>
    public static Text HoverLabel(Plot plot, ChartTheme theme)
    {
        var label = plot.Add.Text(string.Empty, 0, 0);
        label.IsVisible = false;
        Box(label.LabelStyle, theme, 0.95);
        label.LabelStyle.FontSize = theme.TickSize;
        label.LabelStyle.Padding = 4;
        label.Alignment = Alignment.LowerLeft;
        label.OffsetX = 10;
        label.OffsetY = -10;
        return label;
    }

    /// <summary>Show <paramref name="label"/> at <paramref name="location"/> with <paramref name="text"/>.</summary>
    public static void ShowHover(Text label, string text, Coordinates location)
    {
        label.LabelText = text;
        label.LabelStyle.FontName = Fonts.Detect(text);
        label.Location = location;
        label.IsVisible = true;
    }

    /// <summary>A boxed note in a corner of the data area.</summary>
    public static Annotation Note(Plot plot, ChartTheme theme, string text, Alignment alignment)
    {
        var note = plot.Add.Annotation(text, alignment);
        Box(note.LabelStyle, theme, 0.85);
        note.LabelStyle.FontSize = theme.TickSize;
        note.LabelStyle.FontName = Fonts.Detect(text);
        return note;
    }

    /// <summary>Unboxed text in a corner of the data area (warnings, empty-data messages).</summary>
    public static Annotation Caption(Plot plot, ChartTheme theme, string text, Alignment alignment, string color)
    {
        var note = plot.Add.Annotation(text, alignment);
        note.LabelStyle.ForeColor = Color(theme, color);
        note.LabelStyle.BackgroundColor = Colors.Transparent;
        note.LabelStyle.BorderColor = Colors.Transparent;
        note.LabelStyle.ShadowColor = Colors.Transparent;
        note.LabelStyle.FontSize = theme.TickSize;
        note.LabelStyle.FontName = Fonts.Detect(text);
        return note;
    }

    private static void Box(LabelStyle style, ChartTheme theme, double alpha)
    {
        style.ForeColor = Color(theme, theme.Text);
        style.BackgroundColor = Color(theme, theme.PanelBackground).WithAlpha(Alpha(theme, alpha));
        style.BorderColor = Color(theme, theme.PanelOutline);
        style.BorderWidth = 1;
        style.BorderRadius = BoxRadius;
        style.ShadowColor = Colors.Transparent;
    }
}
