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
/// lockstep across panels, a dashed vertical crosshair in every panel, and
/// pointer position reported in data coordinates.
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

    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();
    private bool _syncingLimits;
    private StripChartSpec? _spec;

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
        EnsurePanelCount(strip.Subplots.Count);

        for (var i = 0; i < strip.Subplots.Count; i++)
        {
            var subplot = strip.Subplots[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            var isTop = i == 0;
            var isBottom = i == strip.Subplots.Count - 1;

            plot.Clear();
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
                // SignalXY is the fast path for large, x-sorted series.
                var line = plot.Add.SignalXY(trace.X.ToArray(), trace.Y.ToArray());
                line.Color = Color.FromHex(trace.Color);
                line.LineWidth = 1.5f;
                line.LegendText = trace.Label;
            }

            if (subplot.ZeroLine)
            {
                var zero = plot.Add.HorizontalLine(0);
                zero.LineColor = Colors.Gray.WithAlpha(0.6);
                zero.LineWidth = 1;
                zero.LinePattern = LinePattern.Dotted;
            }

            // The crosshair is added last so it draws on top of the traces.
            panel.Crosshair = plot.Add.VerticalLine(0);
            panel.Crosshair.IsVisible = false;
            panel.Crosshair.LineColor = Colors.White.WithAlpha(0.7);
            panel.Crosshair.LineWidth = 1;
            panel.Crosshair.LinePattern = LinePattern.Dashed;

            plot.Axes.AutoScale();
            panel.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
        foreach (var panel in _panels)
        {
            if (panel.Crosshair is null)
            {
                continue;
            }

            panel.Crosshair.IsVisible = x.HasValue;
            if (x.HasValue)
            {
                panel.Crosshair.X = x.Value;
            }

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
        panel.OnLimitsChanged = (_, _) => SyncXLimitsFrom(panel);

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

    /// <summary>Per-subplot state: the control, its crosshair and the handlers to detach.</summary>
    private sealed class Panel
    {
        public Panel(AvaPlot avaPlot, int index)
        {
            AvaPlot = avaPlot;
            Index = index;
        }

        public AvaPlot AvaPlot { get; }
        public int Index { get; }
        public VerticalLine? Crosshair { get; set; }
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
        public EventHandler<RenderDetails> OnLimitsChanged { get; set; } = (_, _) => { };
    }
}
