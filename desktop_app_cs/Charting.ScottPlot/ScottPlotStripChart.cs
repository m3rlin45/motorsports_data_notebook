using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="StripChartSpec"/>: one
/// <see cref="AvaPlot"/> per subplot stacked in a grid, x-limits kept in
/// lockstep across panels, a dashed vertical crosshair in every panel with
/// each trace's value stacked beside it, and pointer position reported in
/// data coordinates.
/// </summary>
public sealed class ScottPlotStripChart : IChartSurface
{
    // Fixed layout padding so every panel's data area starts and ends at the
    // same pixel columns; without this the y-tick label width would misalign
    // them. Only left/right must match across panels: the bottom panel gets
    // extra room for its tick labels and axis label.
    private const float PadLeft = 72;
    private const float PadRight = 16;
    private const float PadTop = 36;
    private const float PadBottomInner = 14;
    private const float PadBottomLast = 52;

    // Cursor value labels: stacked from the top of the data area, beside the
    // crosshair; flipped to its left when the cursor is in the right quarter
    // of the visible range so they never run off the panel.
    private const float LabelGapPx = 8;
    private const float LabelTopPx = 6;
    private const float LabelLineHeightPx = 16;
    private const float LabelFontSize = 11;
    private const float MarkerSize = 7;
    private const double FlipFraction = 0.75;
    private const double ContextAlpha = 0.3;
    private const double ViewPadFraction = 0.01;

    // Dark theme matching the window chrome (#1e1e1e).
    private static readonly Color FigureBackground = Color.FromHex("#1e1e1e");
    private static readonly Color DataBackground = Color.FromHex("#262626");
    private static readonly Color AxisColor = Color.FromHex("#c8c8c8");
    private static readonly Color GridColor = Color.FromHex("#3c3c3c");
    private static readonly Color PanelBackground = Color.FromHex("#2a2a2a");
    private static readonly Color PanelOutline = Color.FromHex("#505050");
    private static readonly Color TextColor = Color.FromHex("#e8e8e8");
    private static readonly Color CrosshairColor = Color.FromHex("#f5f5f5").WithAlpha(0.85);
    private static readonly Color LabelBackground = Color.FromHex("#1e1e1e").WithAlpha(0.9);
    private static readonly Color MarkerLabelBackground = Color.FromHex("#262626").WithAlpha(0.85);

    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();

    /// <summary>The underlying plots, top to bottom (for tests).</summary>
    internal IReadOnlyList<Plot> Plots => _panels.Select(p => p.AvaPlot.Plot).ToList();
    private bool _syncingLimits;
    private StripChartSpec? _spec;
    private double? _cursor;

    /// <inheritdoc />
    public Control Control => _grid;

    /// <inheritdoc />
    public event Action<double?>? CursorMoved;

    /// <inheritdoc />
    public event Action<ChartPoint>? Clicked;

    /// <inheritdoc />
    public void Render(ChartSpec spec)
    {
        if (spec is not StripChartSpec strip)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(_spec, strip))
        {
            return;
        }

        _spec = strip;
        _cursor = null;
        EnsurePanelCount(strip.Subplots.Count);

        for (var i = 0; i < strip.Subplots.Count; i++)
        {
            var subplot = strip.Subplots[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            var isTop = i == 0;
            var isBottom = i == strip.Subplots.Count - 1;

            plot.Clear();
            panel.Subplot = subplot;
            panel.Cursors.Clear();
            panel.MarkerLabels.Clear();

            ApplyTheme(plot);
            // Named arguments on purpose: PixelPadding's positional order is (left, right, bottom, top).
            plot.Layout.Fixed(new PixelPadding(
                left: PadLeft,
                right: PadRight,
                bottom: isBottom ? PadBottomLast : PadBottomInner,
                top: PadTop));
            plot.Title(subplot.Title, size: 13);
            plot.Axes.Left.Label.Text = subplot.Units;
            plot.Axes.Bottom.Label.Text = isBottom ? strip.XLabel : string.Empty;
            plot.Axes.Bottom.TickLabelStyle.IsVisible = isBottom;
            plot.Legend.IsVisible = isTop;
            plot.Legend.Alignment = Alignment.UpperRight;

            foreach (var trace in subplot.Traces)
            {
                AddTrace(plot, trace);
            }

            foreach (var marker in strip.Markers ?? Array.Empty<XMarkerSpec>())
            {
                var color = Color.FromHex(marker.Color);
                var line = plot.Add.VerticalLine(marker.X);
                line.LineColor = color.WithAlpha(0.7);
                line.LineWidth = 1;
                line.LinePattern = LinePattern.Dotted;
                if (!isBottom)
                {
                    continue; // lines on every panel, labels only next to the x-axis
                }

                var label = plot.Add.Text(marker.Label, marker.X, 0);
                label.LabelFontColor = color;
                label.LabelFontSize = LabelFontSize;
                label.LabelBackgroundColor = MarkerLabelBackground;
                label.LabelPadding = 2;
                label.Alignment = Alignment.LowerLeft;
                label.OffsetX = 4;
                panel.MarkerLabels.Add(new AnchoredLabel(label, marker.X, marker.Row));
            }

            if (subplot.ZeroLine)
            {
                var zero = plot.Add.HorizontalLine(0);
                zero.LineColor = Colors.Gray.WithAlpha(0.6);
                zero.LineWidth = 1;
                zero.LinePattern = LinePattern.Dotted;
            }

            // Cursor decorations are added after the traces so they draw on top.
            foreach (var trace in subplot.Traces)
            {
                var color = Color.FromHex(trace.Color);
                var marker = plot.Add.Marker(0, 0, MarkerShape.FilledCircle, MarkerSize, color);
                marker.IsVisible = false;
                var label = plot.Add.Text(string.Empty, 0, 0);
                label.IsVisible = false;
                label.LabelFontColor = color;
                label.LabelFontSize = LabelFontSize;
                label.LabelBold = true;
                label.LabelBackgroundColor = LabelBackground;
                label.LabelPadding = 2;
                panel.Cursors.Add(new CursorDecoration(trace, marker, label));
            }

            panel.Crosshair = plot.Add.VerticalLine(0);
            panel.Crosshair.IsVisible = false;
            panel.Crosshair.LineColor = CrosshairColor;
            panel.Crosshair.LineWidth = 1;
            panel.Crosshair.LinePattern = LinePattern.Dashed;
            panel.Crosshair.LabelBackgroundColor = PanelBackground;
            panel.Crosshair.LabelFontColor = TextColor;
            panel.Crosshair.LabelFontSize = LabelFontSize;
            panel.ShowsXLabel = isBottom;

            plot.Axes.AutoScale();
            if (strip.InitialView is { } view)
            {
                // A little slack so delimiters sitting exactly on the view edges are visible.
                var pad = ViewPadFraction * (view.Max - view.Min);
                plot.Axes.SetLimitsX(view.Min - pad, view.Max + pad);
            }

            AnchorMarkerLabels(panel);
            panel.AvaPlot.Refresh();
        }
    }

    /// <summary>
    /// Add a trace as one or more SignalXY plottables (the fast path for large,
    /// x-sorted series): the emphasised range at full strength with the legend
    /// entry, the context on either side faded and unlabelled.
    /// </summary>
    private static void AddTrace(Plot plot, TraceSpec trace)
    {
        var xs = trace.X.ToArray();
        var ys = trace.Y.ToArray();
        var color = Color.FromHex(trace.Color);

        var start = 0;
        var end = xs.Length;
        if (trace.Emphasis is { } emphasis)
        {
            start = LowerBound(xs, emphasis.Min);
            end = UpperBound(xs, emphasis.Max);
        }

        void Add(int from, int to, double alpha, string legend)
        {
            if (to - from < 2)
            {
                return;
            }

            var line = plot.Add.SignalXY(xs[from..to], ys[from..to]);
            line.Color = color.WithAlpha(alpha);
            line.LineWidth = 1.5f;
            line.LegendText = legend;
        }

        // Overlap the splits by one sample so the line has no gap at the boundary.
        Add(0, Math.Min(start + 1, xs.Length), ContextAlpha, string.Empty);
        Add(start, end, 1, trace.Label);
        Add(Math.Max(end - 1, 0), xs.Length, ContextAlpha, string.Empty);
    }

    /// <summary>Index of the first element ≥ <paramref name="value"/>.</summary>
    private static int LowerBound(double[] xs, double value)
    {
        var i = Array.BinarySearch(xs, value);
        if (i >= 0)
        {
            while (i > 0 && xs[i - 1] == value)
            {
                i--;
            }

            return i;
        }

        return ~i;
    }

    /// <summary>Index one past the last element ≤ <paramref name="value"/>.</summary>
    private static int UpperBound(double[] xs, double value)
    {
        var i = Array.BinarySearch(xs, value);
        if (i >= 0)
        {
            while (i < xs.Length - 1 && xs[i + 1] == value)
            {
                i++;
            }

            return i + 1;
        }

        return ~i;
    }

    /// <summary>
    /// Pin the lap-boundary labels to the bottom of the data area at their x.
    /// Text positions are data coordinates, so the y must follow the visible
    /// range; rows stack upward in pixels.
    /// </summary>
    private static void AnchorMarkerLabels(Panel panel)
    {
        var yBottom = panel.AvaPlot.Plot.Axes.Left.Range.Min;
        foreach (var anchored in panel.MarkerLabels)
        {
            anchored.Label.Location = new Coordinates(anchored.X, yBottom);
            anchored.Label.OffsetY = -(LabelTopPx + anchored.Row * LabelLineHeightPx);
        }
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
        _cursor = x;
        foreach (var panel in _panels)
        {
            UpdateCursor(panel);
            panel.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var panel in _panels)
        {
            Detach(panel);
        }

        _panels.Clear();
        _grid.Children.Clear();
        _grid.RowDefinitions.Clear();
    }

    private static void ApplyTheme(Plot plot)
    {
        plot.FigureBackground.Color = FigureBackground;
        plot.DataBackground.Color = DataBackground;
        plot.Axes.Color(AxisColor);
        plot.Axes.Title.Label.ForeColor = TextColor;
        plot.Grid.MajorLineColor = GridColor;
        plot.Legend.BackgroundColor = PanelBackground;
        plot.Legend.OutlineColor = PanelOutline;
        plot.Legend.FontColor = TextColor;
    }

    /// <summary>Position the crosshair, markers and value labels of one panel for the current cursor.</summary>
    private void UpdateCursor(Panel panel)
    {
        if (panel.Crosshair is null || panel.Subplot is null || _spec is null)
        {
            return;
        }

        var x = _cursor;
        panel.Crosshair.IsVisible = x.HasValue;
        if (!x.HasValue)
        {
            foreach (var decoration in panel.Cursors)
            {
                decoration.Marker.IsVisible = false;
                decoration.Label.IsVisible = false;
            }

            return;
        }

        var plot = panel.AvaPlot.Plot;
        var xRange = plot.Axes.Bottom.Range;
        var yTop = plot.Axes.Left.Range.Max;
        var flip = x.Value > xRange.Min + FlipFraction * (xRange.Max - xRange.Min);
        var format = "F" + panel.Subplot.Decimals.ToString(CultureInfo.InvariantCulture);

        panel.Crosshair.X = x.Value;
        panel.Crosshair.LabelText = panel.ShowsXLabel ? _spec.FormatX(x.Value) : string.Empty;

        var slot = 0;
        foreach (var decoration in panel.Cursors)
        {
            var y = decoration.Trace.ValueAt(x.Value);
            var visible = !double.IsNaN(y);
            decoration.Marker.IsVisible = visible;
            decoration.Label.IsVisible = visible;
            if (!visible)
            {
                continue;
            }

            decoration.Marker.Location = new Coordinates(x.Value, y);

            var text = y.ToString(format, CultureInfo.InvariantCulture);
            if (panel.Subplot.Units.Length > 0)
            {
                text += " " + panel.Subplot.Units;
            }

            decoration.Label.LabelText = text;
            decoration.Label.Location = new Coordinates(x.Value, yTop);
            decoration.Label.Alignment = flip ? Alignment.UpperRight : Alignment.UpperLeft;
            decoration.Label.OffsetX = flip ? -LabelGapPx : LabelGapPx;
            decoration.Label.OffsetY = LabelTopPx + slot * LabelLineHeightPx;
            slot++;
        }
    }

    private void EnsurePanelCount(int count)
    {
        while (_panels.Count > count)
        {
            var last = _panels[^1];
            Detach(last);
            _grid.Children.Remove(last.AvaPlot);
            _panels.RemoveAt(_panels.Count - 1);
        }

        while (_panels.Count < count)
        {
            var index = _panels.Count;
            var avaPlot = new AvaPlot();
            var panel = new Panel(avaPlot, index);
            Attach(panel);
            Grid.SetRow(avaPlot, index);
            _grid.Children.Add(avaPlot);
            _panels.Add(panel);
        }

        _grid.RowDefinitions.Clear();
        for (var i = 0; i < count; i++)
        {
            _grid.RowDefinitions.Add(new RowDefinition(1, GridUnitType.Star));
        }
    }

    private void Attach(Panel panel)
    {
        var avaPlot = panel.AvaPlot;
        panel.OnPointerMoved = (_, e) => OnPointerMoved(panel, e);
        panel.OnPointerExited = (_, _) => CursorMoved?.Invoke(null);
        panel.OnPointerPressed = (_, e) => OnPointerPressed(panel, e);
        panel.OnLimitsChanged = (_, _) => OnLimitsChanged(panel);

        DataAreaInput.Install(avaPlot.UserInputProcessor);
        avaPlot.PointerMoved += panel.OnPointerMoved;
        avaPlot.PointerExited += panel.OnPointerExited;
        avaPlot.PointerPressed += panel.OnPointerPressed;
        avaPlot.Plot.RenderManager.AxisLimitsChanged += panel.OnLimitsChanged;
    }

    private static void Detach(Panel panel)
    {
        var avaPlot = panel.AvaPlot;
        avaPlot.PointerMoved -= panel.OnPointerMoved;
        avaPlot.PointerExited -= panel.OnPointerExited;
        avaPlot.PointerPressed -= panel.OnPointerPressed;
#pragma warning disable CS8601 // ScottPlot declares the event field non-nullable; unsubscribing may null it
        avaPlot.Plot.RenderManager.AxisLimitsChanged -= panel.OnLimitsChanged;
#pragma warning restore CS8601
    }

    private Coordinates ToCoordinates(Panel panel, PointerEventArgs e)
    {
        var position = e.GetPosition(panel.AvaPlot);
        var scale = panel.AvaPlot.DisplayScale;
        var pixel = new Pixel((float)(position.X * scale), (float)(position.Y * scale));
        return panel.AvaPlot.Plot.GetCoordinates(pixel);
    }

    private void OnPointerMoved(Panel panel, PointerEventArgs e)
    {
        var coordinates = ToCoordinates(panel, e);
        SetCursor(coordinates.X);
        CursorMoved?.Invoke(coordinates.X);
    }

    private void OnPointerPressed(Panel panel, PointerPressedEventArgs e)
    {
        var coordinates = ToCoordinates(panel, e);
        Clicked?.Invoke(new ChartPoint(coordinates.X, coordinates.Y, panel.Index));
    }

    private void OnLimitsChanged(Panel source)
    {
        SyncXLimitsFrom(source);

        // The value labels hang from the top of the data area and the lap
        // labels from the bottom, so a pan or zoom moves them. Re-anchor; the
        // extra render is a no-op if nothing moved.
        AnchorMarkerLabels(source);
        if (_cursor.HasValue)
        {
            UpdateCursor(source);
        }

        source.AvaPlot.Refresh();
    }

    private void SyncXLimitsFrom(Panel source)
    {
        if (_syncingLimits)
        {
            return;
        }

        _syncingLimits = true;
        try
        {
            var range = source.AvaPlot.Plot.Axes.Bottom.Range;
            foreach (var panel in _panels)
            {
                if (ReferenceEquals(panel, source))
                {
                    continue;
                }

                var target = panel.AvaPlot.Plot.Axes.Bottom.Range;
                if (target.Min == range.Min && target.Max == range.Max)
                {
                    continue;
                }

                panel.AvaPlot.Plot.Axes.SetLimitsX(range.Min, range.Max);
                panel.AvaPlot.Refresh();
            }
        }
        finally
        {
            _syncingLimits = false;
        }
    }

    /// <summary>The marker and value label that follow the cursor along one trace.</summary>
    private sealed record CursorDecoration(TraceSpec Trace, Marker Marker, Text Label);

    /// <summary>A label pinned to the data-area edge at a fixed x.</summary>
    private sealed record AnchoredLabel(Text Label, double X, int Row);

    /// <summary>Per-subplot state: the control, its cursor plottables and the handlers to detach.</summary>
    private sealed class Panel
    {
        public Panel(AvaPlot avaPlot, int index)
        {
            AvaPlot = avaPlot;
            Index = index;
        }

        public AvaPlot AvaPlot { get; }
        public int Index { get; }
        public SubplotSpec? Subplot { get; set; }
        public VerticalLine? Crosshair { get; set; }
        public bool ShowsXLabel { get; set; }
        public List<CursorDecoration> Cursors { get; } = new();
        public List<AnchoredLabel> MarkerLabels { get; } = new();
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
        public EventHandler<RenderDetails> OnLimitsChanged { get; set; } = (_, _) => { };
    }
}
