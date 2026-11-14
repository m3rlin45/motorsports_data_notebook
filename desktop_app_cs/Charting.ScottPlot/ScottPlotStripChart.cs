using System.Globalization;
using System.Runtime.InteropServices;
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
/// lockstep across panels, and a crosshair with each trace's value beside it
/// drawn by a <see cref="StripCursorOverlay"/> above every panel.
/// </summary>
/// <remarks>
/// Interaction: the crosshair follows the pointer; a click pins it (Esc
/// releases it), Left/Right step the pinned cursor one sample of the
/// reference trace (ten with Shift), Up/Down zoom about it, and Home or a
/// double-click returns to the initial view. Dragging and the wheel pan and
/// zoom as set up by <see cref="DataAreaInput"/>.
/// </remarks>
public sealed class ScottPlotStripChart : ScottPlotSurface<StripChartSpec>
{
    // Fixed layout padding so every panel's data area starts and ends at the
    // same pixel columns; without this the y-tick label width would misalign
    // them. Only left/right must match across panels: the bottom panel gets
    // extra room for its tick labels and axis label. The title is a tag
    // inside the data area, so no band is reserved above it.
    private const float PadLeft = 64;
    private const float PadRight = 16;
    private const float PadTop = 8;
    private const float PadBottomInner = 8;
    private const float PadBottomLast = 46;

    private const float MarkerLabelTopPx = 6;
    private const float MarkerLabelLineHeightPx = 16;
    private const float TraceWidth = 1.25f;
    private const float ReferenceTraceWidth = 2f;
    private const double ContextAlpha = 0.3;
    private const double ViewPadFraction = 0.01;
    private const double KeyZoomFraction = 0.25;
    private const double ClickSlopPx = 4;

    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();
    private bool _syncingLimits;
    private double? _cursor;
    private bool _pinned;

    /// <summary>Create the surface.</summary>
    public ScottPlotStripChart()
    {
        // Keys arrive here from whichever panel has the focus (a click on a plot focuses it). The grid
        // itself must not take the focus: a plot that loses it mid-press abandons the drag in progress.
        _grid.KeyDown += OnKeyDown;
    }

    /// <inheritdoc />
    public override Control Control => _grid;

    /// <summary>The underlying plots, top to bottom (for tests).</summary>
    internal IReadOnlyList<Plot> Plots => _panels.Select(p => p.AvaPlot.Plot).ToList();

    /// <summary>The cursor overlays, top to bottom (for tests).</summary>
    internal IReadOnlyList<StripCursorOverlay> Overlays => _panels.Select(p => p.Overlay).ToList();

    /// <summary>The cursor position, or null when hidden.</summary>
    public double? Cursor => _cursor;

    /// <summary>Whether the cursor is pinned (it stays where it was put instead of following the pointer).</summary>
    public bool IsCursorPinned => _pinned;

    /// <inheritdoc />
    protected override void Build(StripChartSpec spec, StripChartSpec? previous)
    {
        // A spec that shows the same thing (a theme change, a lap or channel ticked) keeps the user's
        // pan/zoom and cursor; a new session, a new corner or a different x-axis starts from the initial view.
        var sameView = previous is not null
            && _panels.Count > 0
            && previous.XAxis == spec.XAxis
            && (ReferenceEquals(previous, spec) || (spec.ViewKey is not null && spec.ViewKey == previous.ViewKey));
        XRange? keep = null;
        if (sameView)
        {
            var range = _panels[0].AvaPlot.Plot.Axes.Bottom.Range;
            keep = new XRange(range.Min, range.Max);
        }
        else
        {
            _cursor = null;
            _pinned = false;
        }

        EnsurePanelCount(spec.Subplots.Count);

        for (var i = 0; i < spec.Subplots.Count; i++)
        {
            var subplot = spec.Subplots[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            var isTop = i == 0;
            var isBottom = i == spec.Subplots.Count - 1;

            plot.Clear();
            panel.Subplot = subplot;
            panel.MarkerLabels.Clear();
            panel.ShowsXLabel = isBottom;

            PlotStyle.Apply(plot, Theme);
            // Named arguments on purpose: PixelPadding's positional order is (left, right, bottom, top).
            var padding = new PixelPadding(left: PadLeft, right: PadRight, bottom: isBottom ? PadBottomLast : PadBottomInner, top: PadTop);
            plot.Layout.Fixed(padding);
            plot.Axes.Title.Label.Text = string.Empty;
            plot.Axes.Left.Label.Text = string.Empty;
            plot.Axes.Bottom.Label.Text = isBottom ? spec.XLabel : string.Empty;
            plot.Axes.Bottom.TickLabelStyle.IsVisible = isBottom;
            plot.Legend.IsVisible = isTop && spec.Legend;
            plot.Legend.Alignment = Alignment.UpperRight;

            foreach (var trace in subplot.Traces)
            {
                AddTrace(plot, trace);
            }

            foreach (var marker in spec.Markers ?? Array.Empty<XMarkerSpec>())
            {
                var color = ColorOf(marker.Color);
                var line = plot.Add.VerticalLine(marker.X);
                line.LineColor = color.WithAlpha(PlotStyle.Alpha(Theme, 0.7));
                line.LineWidth = 1;
                line.LinePattern = LinePattern.Dotted;
                if (!isBottom || marker.Label.Length == 0)
                {
                    continue; // lines on every panel, labels only next to the x-axis
                }

                var label = plot.Add.Text(marker.Label, marker.X, 0);
                label.LabelFontColor = ColorOf(Theme.Axis);
                label.LabelFontSize = Theme.TickSize;
                label.LabelFontName = Fonts.Detect(marker.Label);
                label.LabelBackgroundColor = ColorOf(Theme.DataBackground).WithAlpha(PlotStyle.Alpha(Theme, 0.85));
                label.LabelPadding = 2;
                // A boundary at the end of the initial view puts its label on the inside, where it is not clipped.
                var atEnd = spec.InitialView is { } view && marker.X >= view.Max - 0.02 * (view.Max - view.Min);
                label.Alignment = atEnd ? Alignment.LowerRight : Alignment.LowerLeft;
                label.OffsetX = atEnd ? -4 : 4;
                panel.MarkerLabels.Add(new AnchoredLabel(label, marker.X, marker.Row));
            }

            if (subplot.ZeroLine)
            {
                var zero = plot.Add.HorizontalLine(0);
                zero.LineColor = ColorOf(Theme.Axis).WithAlpha(0.6);
                zero.LineWidth = 1;
                zero.LinePattern = LinePattern.Dotted;
            }

            Fit(plot);
            ApplyXView(plot, keep ?? spec.InitialView, padded: keep is null);
            AnchorMarkerLabels(panel);
            PlotStyle.FitFonts(plot);

            panel.Overlay.Configure(TitleTag(subplot), Theme, padding);
            panel.AvaPlot.Refresh();
        }

        UpdateOverlays();
    }

    /// <summary>Fit the data, leaving headroom above it for the title tag that sits inside the data area.</summary>
    private static void Fit(Plot plot)
    {
        plot.Axes.Margins(left: 0, right: 0, bottom: 0.06, top: 0.2);
        plot.Axes.AutoScale();
    }

    /// <summary>"Speed · km/h": the panel's name and unit as one tag inside the data area.</summary>
    private static string TitleTag(SubplotSpec subplot) =>
        subplot.Units.Length > 0 ? $"{subplot.Title} · {subplot.Units}" : subplot.Title;

    private static void ApplyXView(Plot plot, XRange? view, bool padded)
    {
        if (view is not { } v)
        {
            return;
        }

        // A little slack so delimiters sitting exactly on the view edges are visible.
        var pad = padded ? ViewPadFraction * (v.Max - v.Min) : 0;
        plot.Axes.SetLimitsX(v.Min - pad, v.Max + pad);
    }

    /// <summary>
    /// Add a trace as up to three SignalXY plottables (the fast path for
    /// large, x-sorted series) over the same two arrays: the emphasised range
    /// at full strength with the legend entry, the context on either side
    /// faded and unlabelled. Each draws its own index range, so nothing is
    /// copied when the spec's memory is a whole array.
    /// </summary>
    private void AddTrace(Plot plot, TraceSpec trace)
    {
        var xs = AsArray(trace.X);
        var ys = AsArray(trace.Y);
        var color = ColorOf(trace.Color);

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

            var line = plot.Add.SignalXY(xs, ys);
            line.MinRenderIndex = from;
            line.MaxRenderIndex = to - 1;
            line.Color = color.WithAlpha(alpha);
            line.LineWidth = (trace.Reference ? ReferenceTraceWidth : TraceWidth) * Theme.LineScale;
            line.LinePattern = trace.Dashed ? LinePattern.Dashed : LinePattern.Solid;
            line.LegendText = legend;
        }

        // Overlap the splits by one sample so the line has no gap at the boundary.
        Add(0, Math.Min(start + 1, xs.Length), ContextAlpha, string.Empty);
        Add(start, end, 1, trace.Label);
        Add(Math.Max(end - 1, 0), xs.Length, ContextAlpha, string.Empty);
    }

    /// <summary>The array behind <paramref name="memory"/> when it spans one exactly, else a copy.</summary>
    private static double[] AsArray(ReadOnlyMemory<double> memory) =>
        MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array is { } array && segment.Offset == 0 && segment.Count == array.Length
            ? array
            : memory.ToArray();

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
            anchored.Label.OffsetY = -(MarkerLabelTopPx + anchored.Row * MarkerLabelLineHeightPx);
        }
    }

    /// <inheritdoc />
    public override void SetCursor(double? x)
    {
        _cursor = x;
        if (x is null)
        {
            _pinned = false;
        }

        UpdateOverlays();
    }

    /// <inheritdoc />
    public override void ResetView()
    {
        if (Spec is not { } spec)
        {
            return;
        }

        foreach (var panel in _panels)
        {
            var plot = panel.AvaPlot.Plot;
            Fit(plot);
            ApplyXView(plot, spec.InitialView, padded: true);
            AnchorMarkerLabels(panel);
            panel.AvaPlot.Refresh();
        }

        UpdateOverlays();
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _grid.KeyDown -= OnKeyDown;
        foreach (var panel in _panels)
        {
            Detach(panel);
        }

        _panels.Clear();
        _grid.Children.Clear();
        _grid.RowDefinitions.Clear();
        base.Dispose();
    }

    /// <summary>Recompute every panel's read-outs for the current cursor and repaint the overlays (not the plots).</summary>
    private void UpdateOverlays()
    {
        foreach (var panel in _panels)
        {
            UpdateOverlay(panel);
        }
    }

    private void UpdateOverlay(Panel panel)
    {
        if (panel.Subplot is null || Spec is null || _cursor is not { } x)
        {
            panel.Overlay.SetCursor(null, false, string.Empty, Array.Empty<StripCursorOverlay.Readout>());
            return;
        }

        var format = "F" + panel.Subplot.Decimals.ToString(CultureInfo.InvariantCulture);
        var readouts = new List<StripCursorOverlay.Readout>(panel.Subplot.Traces.Count);
        foreach (var trace in panel.Subplot.Traces)
        {
            var y = trace.ValueAt(x);
            if (double.IsNaN(y))
            {
                continue;
            }

            var text = y.ToString(format, CultureInfo.InvariantCulture);
            if (panel.Subplot.Units.Length > 0)
            {
                text += " " + panel.Subplot.Units;
            }

            if (trace.ShortLabel.Length > 0)
            {
                text = trace.ShortLabel + "  " + text;
            }

            readouts.Add(new StripCursorOverlay.Readout(y, Theme.Resolve(trace.Color), trace.Dashed, text));
        }

        panel.Overlay.SetCursor(x, _pinned, panel.ShowsXLabel ? Spec.FormatX(x) : string.Empty, readouts);
    }

    private void EnsurePanelCount(int count)
    {
        while (_panels.Count > count)
        {
            var last = _panels[^1];
            Detach(last);
            _grid.Children.Remove(last.AvaPlot);
            _grid.Children.Remove(last.Overlay);
            _panels.RemoveAt(_panels.Count - 1);
        }

        while (_panels.Count < count)
        {
            var index = _panels.Count;
            var avaPlot = new AvaPlot();
            var panel = new Panel(avaPlot, new StripCursorOverlay(avaPlot.Plot), index);
            Attach(panel);
            Grid.SetRow(avaPlot, index);
            Grid.SetRow(panel.Overlay, index);
            _grid.Children.Add(avaPlot);
            _grid.Children.Add(panel.Overlay);
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
        panel.OnPointerExited = (_, _) => OnPointerExited();
        panel.OnPointerPressed = (_, e) => OnPointerPressed(panel, e);
        panel.OnPointerReleased = (_, e) => OnPointerReleased(panel, e);
        panel.OnDoubleTapped = (_, _) => OnDoubleTapped();
        panel.OnLimitsChanged = (_, _) => OnLimitsChanged(panel);

        DataAreaInput.Install(avaPlot.UserInputProcessor);
        avaPlot.PointerMoved += panel.OnPointerMoved;
        avaPlot.PointerExited += panel.OnPointerExited;
        avaPlot.PointerPressed += panel.OnPointerPressed;
        avaPlot.PointerReleased += panel.OnPointerReleased;
        avaPlot.DoubleTapped += panel.OnDoubleTapped;
        avaPlot.Plot.RenderManager.AxisLimitsChanged += panel.OnLimitsChanged;
    }

    private static void Detach(Panel panel)
    {
        var avaPlot = panel.AvaPlot;
        avaPlot.PointerMoved -= panel.OnPointerMoved;
        avaPlot.PointerExited -= panel.OnPointerExited;
        avaPlot.PointerPressed -= panel.OnPointerPressed;
        avaPlot.PointerReleased -= panel.OnPointerReleased;
        avaPlot.DoubleTapped -= panel.OnDoubleTapped;
#pragma warning disable CS8601 // ScottPlot declares the event field non-nullable; unsubscribing may null it
        avaPlot.Plot.RenderManager.AxisLimitsChanged -= panel.OnLimitsChanged;
#pragma warning restore CS8601
    }

    private void OnPointerMoved(Panel panel, PointerEventArgs e)
    {
        if (_pinned)
        {
            return; // a pinned cursor stays put; the pointer is free to go and tick a lap
        }

        var x = ToCoordinates(panel.AvaPlot, e).X;
        _cursor = x;
        UpdateOverlays();
        RaiseCursorMoved(x);
    }

    private void OnPointerExited()
    {
        if (_pinned)
        {
            return;
        }

        _cursor = null;
        UpdateOverlays();
        RaiseCursorMoved(null);
    }

    private void OnPointerPressed(Panel panel, PointerPressedEventArgs e)
    {
        var point = e.GetCurrentPoint(panel.AvaPlot);
        panel.PressedAt = point.Properties.IsLeftButtonPressed ? point.Position : null;
        var at = ToCoordinates(panel.AvaPlot, e);
        RaiseClicked(new ChartPoint(at.X, at.Y, panel.Index));
    }

    private void OnPointerReleased(Panel panel, PointerReleasedEventArgs e)
    {
        if (panel.PressedAt is not { } pressed)
        {
            return;
        }

        panel.PressedAt = null;
        var released = e.GetPosition(panel.AvaPlot);
        if (Math.Abs(released.X - pressed.X) > ClickSlopPx || Math.Abs(released.Y - pressed.Y) > ClickSlopPx)
        {
            return; // a drag (pan), not a click
        }

        PinCursor(ToCoordinates(panel.AvaPlot, e).X);
    }

    private void OnDoubleTapped()
    {
        UnpinCursor();
        ResetView();
    }

    /// <summary>Put the cursor at <paramref name="x"/> and keep it there until <see cref="UnpinCursor"/>.</summary>
    public void PinCursor(double x)
    {
        _cursor = x;
        _pinned = true;
        UpdateOverlays();
        RaiseCursorMoved(x);
    }

    /// <summary>Let the cursor follow the pointer again (hidden until the pointer next moves over the chart).</summary>
    public void UnpinCursor()
    {
        if (!_pinned)
        {
            return;
        }

        _pinned = false;
        _cursor = null;
        UpdateOverlays();
        RaiseCursorMoved(null);
    }

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (Spec is null || _panels.Count == 0)
        {
            return;
        }

        var shift = e.KeyModifiers.HasFlag(KeyModifiers.Shift);
        switch (e.Key)
        {
            case Key.Left:
                StepCursor(shift ? -10 : -1);
                break;
            case Key.Right:
                StepCursor(shift ? 10 : 1);
                break;
            case Key.Up:
                ZoomAboutCursor(1 - KeyZoomFraction);
                break;
            case Key.Down:
                ZoomAboutCursor(1 / (1 - KeyZoomFraction));
                break;
            case Key.Home:
                ResetView();
                break;
            case Key.Escape when _pinned:
                UnpinCursor();
                break;
            default:
                return;
        }

        e.Handled = true;
    }

    /// <summary>
    /// Move the cursor <paramref name="samples"/> samples along the reference
    /// trace (the first trace of the first panel that has one), pinning it;
    /// from no cursor it starts at the middle of the view. The view scrolls
    /// to keep the cursor visible.
    /// </summary>
    public void StepCursor(int samples)
    {
        var reference = _panels.Select(p => p.Subplot).Where(s => s is not null).SelectMany(s => s!.Traces).FirstOrDefault(t => t.X.Length > 0);
        if (reference is null)
        {
            return;
        }

        var xs = reference.X.Span;
        var range = _panels[0].AvaPlot.Plot.Axes.Bottom.Range;
        var from = _cursor ?? (range.Min + range.Max) / 2;
        var index = xs.BinarySearch(from);
        if (index < 0)
        {
            index = ~index;
            if (index > 0 && (index == xs.Length || from - xs[index - 1] <= xs[index] - from))
            {
                index--; // the nearer of the two neighbours
            }
        }

        index = Math.Clamp(index + samples, 0, xs.Length - 1);
        var x = xs[index];
        if (x < range.Min || x > range.Max)
        {
            var shift = x < range.Min ? x - range.Min : x - range.Max;
            SetXLimits(range.Min + shift, range.Max + shift);
        }

        PinCursor(x);
    }

    /// <summary>Scale the visible x span by <paramref name="factor"/> about the cursor (the view centre without one).</summary>
    public void ZoomAboutCursor(double factor)
    {
        var range = _panels[0].AvaPlot.Plot.Axes.Bottom.Range;
        var centre = _cursor is { } c && c >= range.Min && c <= range.Max ? c : (range.Min + range.Max) / 2;
        SetXLimits(centre - (centre - range.Min) * factor, centre + (range.Max - centre) * factor);
    }

    private void SetXLimits(double min, double max)
    {
        foreach (var panel in _panels)
        {
            panel.AvaPlot.Plot.Axes.SetLimitsX(min, max);
            panel.AvaPlot.Refresh();
        }

        UpdateOverlays();
    }

    private void OnLimitsChanged(Panel source)
    {
        SyncXLimitsFrom(source);

        // The lap labels hang from the bottom of the data area and the overlay maps data to pixels
        // through the limits, so a pan or zoom moves both. The extra plot render is a no-op if nothing moved.
        AnchorMarkerLabels(source);
        source.Overlay.InvalidateVisual();
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
                panel.Overlay.InvalidateVisual();
                panel.AvaPlot.Refresh();
            }
        }
        finally
        {
            _syncingLimits = false;
        }
    }

    /// <summary>A label pinned to the data-area edge at a fixed x.</summary>
    private sealed record AnchoredLabel(Text Label, double X, int Row);

    /// <summary>Per-subplot state: the plot control, its overlay and the handlers to detach.</summary>
    private sealed class Panel
    {
        public Panel(AvaPlot avaPlot, StripCursorOverlay overlay, int index)
        {
            AvaPlot = avaPlot;
            Overlay = overlay;
            Index = index;
        }

        public AvaPlot AvaPlot { get; }
        public StripCursorOverlay Overlay { get; }
        public int Index { get; }
        public SubplotSpec? Subplot { get; set; }
        public bool ShowsXLabel { get; set; }
        public Point? PressedAt { get; set; }
        public List<AnchoredLabel> MarkerLabels { get; } = new();
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
        public EventHandler<PointerReleasedEventArgs> OnPointerReleased { get; set; } = (_, _) => { };
        public EventHandler<TappedEventArgs> OnDoubleTapped { get; set; } = (_, _) => { };
        public EventHandler<RenderDetails> OnLimitsChanged { get; set; } = (_, _) => { };
    }
}
