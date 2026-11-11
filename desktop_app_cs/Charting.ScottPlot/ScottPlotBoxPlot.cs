using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="BoxPlotSpec"/>: stacked panels
/// of boxes with whiskers, a dashed mean line and flier markers, shaded
/// category bands, and a hover tooltip with the box statistics.
/// </summary>
public sealed class ScottPlotBoxPlot : IChartSurface
{
    private const float PadLeft = 72;
    private const float PadRight = 16;
    private const float PadTop = 30;

    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();
    private BoxPlotSpec? _spec;
    private ChartTheme _theme = ChartTheme.Dark;

    /// <inheritdoc />
    public Control Control => _grid;

    /// <inheritdoc />
    public event Action<double?>? CursorMoved;

    /// <inheritdoc />
    public event Action<ChartPoint>? Clicked;

    /// <inheritdoc />
    public ChartTheme Theme
    {
        get => _theme;
        set
        {
            if (ReferenceEquals(_theme, value))
            {
                return;
            }

            _theme = value;
            if (_spec is { } current)
            {
                _spec = null;
                Render(current);
            }
        }
    }

    /// <inheritdoc />
    public void Render(ChartSpec spec)
    {
        if (spec is not BoxPlotSpec box)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(_spec, box))
        {
            return;
        }

        _spec = box;
        EnsurePanelCount(box.Panels.Count);
        var text = Color.FromHex(_theme.Text);

        for (var i = 0; i < box.Panels.Count; i++)
        {
            var panelSpec = box.Panels[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            var isBottom = i == box.Panels.Count - 1;

            plot.Clear();
            panel.Boxes.Clear();
            ScottPlotStripChart.ApplyTheme(plot, _theme);
            plot.Layout.Fixed(new PixelPadding(left: PadLeft, right: PadRight, bottom: isBottom ? 90 : 14, top: PadTop));
            plot.Title(i == 0 ? box.Title : panelSpec.Title, size: 12);
            plot.Axes.Left.Label.Text = panelSpec.YLabel;
            plot.Axes.Bottom.TickLabelStyle.IsVisible = isBottom;
            plot.Legend.IsVisible = i == 0 && box.Legend.Count > 0;
            plot.Legend.Alignment = Alignment.UpperRight;

            foreach (var band in box.Bands)
            {
                var span = plot.Add.HorizontalSpan(band.Start, band.End);
                span.FillColor = Color.FromHex(band.Color).WithAlpha(band.Alpha);
                span.LineWidth = 0;
            }

            if (panelSpec.ZeroLine)
            {
                var zero = plot.Add.HorizontalLine(0);
                zero.LineColor = text.WithAlpha(0.3);
                zero.LineWidth = 0.5f;
            }

            foreach (var b in panelSpec.Boxes)
            {
                // One plottable per box so each keeps its own fill colour.
                var shape = new Box
                {
                    Position = b.Position,
                    BoxMin = b.Q1,
                    BoxMiddle = b.Median,
                    BoxMax = b.Q3,
                    WhiskerMin = b.WhiskerLow,
                    WhiskerMax = b.WhiskerHigh,
                    Width = b.Width,
                };
                var plottable = plot.Add.Box(shape);
                plottable.FillColor = Color.FromHex(b.Color).WithAlpha(0.7);
                plottable.LineColor = text;
                plottable.LineWidth = 1;
                panel.Boxes.Add(b);
            }

            foreach (var b in panelSpec.Boxes)
            {
                // Median in yellow, mean dashed, fliers as small circles — the matplotlib style of the Python app.
                var median = plot.Add.Line(b.Position - b.Width / 2, b.Median, b.Position + b.Width / 2, b.Median);
                median.Color = Colors.Yellow;
                median.LineWidth = 1.5f;
                var mean = plot.Add.Line(b.Position - b.Width / 2, b.Mean, b.Position + b.Width / 2, b.Mean);
                mean.Color = text;
                mean.LineWidth = 1.5f;
                mean.LinePattern = LinePattern.Dashed;
                if (b.Fliers.Length > 0)
                {
                    var fliers = plot.Add.Markers(Enumerable.Repeat(b.Position, b.Fliers.Length).ToArray(), b.Fliers, MarkerShape.OpenCircle, 5, text.WithAlpha(0.7));
                    fliers.MarkerLineWidth = 1;
                }
            }

            foreach (var (label, color) in box.Legend)
            {
                if (i == 0)
                {
                    plot.Legend.ManualItems.Add(new LegendItem { LabelText = label, FillColor = Color.FromHex(color).WithAlpha(0.7) });
                }
            }

            plot.Axes.Bottom.SetTicks(box.CategoryPositions, box.CategoryLabels);
            plot.Axes.Bottom.TickLabelStyle.Rotation = -45;
            plot.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleRight;

            panel.Hover = plot.Add.Text(string.Empty, 0, 0);
            panel.Hover.IsVisible = false;
            panel.Hover.LabelFontColor = text;
            panel.Hover.LabelFontSize = 10;
            panel.Hover.LabelFontName = Fonts.Monospace;
            panel.Hover.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.95);
            panel.Hover.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
            panel.Hover.LabelPadding = 4;
            panel.Hover.Alignment = Alignment.LowerLeft;
            panel.Hover.OffsetX = 12;
            panel.Hover.OffsetY = -12;

            plot.Axes.AutoScale();
            var n = box.CategoryPositions.Length;
            if (n > 0)
            {
                plot.Axes.SetLimitsX(box.CategoryPositions[0] - 0.75, box.CategoryPositions[n - 1] + 0.75);
            }

            panel.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
        // Categories, not samples.
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
            var avaPlot = new AvaPlot();
            var panel = new Panel(avaPlot, _panels.Count);
            Attach(panel);
            Grid.SetRow(avaPlot, panel.Index);
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
        panel.OnPointerMoved = (_, e) => OnPointerMoved(panel, e);
        panel.OnPointerExited = (_, _) => OnPointerExited(panel);
        panel.OnPointerPressed = (_, e) =>
        {
            var c = ToCoordinates(panel, e);
            Clicked?.Invoke(new ChartPoint(c.X, c.Y, panel.Index));
        };
        panel.AvaPlot.PointerMoved += panel.OnPointerMoved;
        panel.AvaPlot.PointerExited += panel.OnPointerExited;
        panel.AvaPlot.PointerPressed += panel.OnPointerPressed;
        DataAreaInput.Install(panel.AvaPlot.UserInputProcessor);
    }

    private static void Detach(Panel panel)
    {
        panel.AvaPlot.PointerMoved -= panel.OnPointerMoved;
        panel.AvaPlot.PointerExited -= panel.OnPointerExited;
        panel.AvaPlot.PointerPressed -= panel.OnPointerPressed;
    }

    private static Coordinates ToCoordinates(Panel panel, PointerEventArgs e)
    {
        var position = e.GetPosition(panel.AvaPlot);
        var scale = panel.AvaPlot.DisplayScale;
        return panel.AvaPlot.Plot.GetCoordinates(new Pixel((float)(position.X * scale), (float)(position.Y * scale)));
    }

    private void OnPointerMoved(Panel panel, PointerEventArgs e)
    {
        var c = ToCoordinates(panel, e);
        CursorMoved?.Invoke(c.X);
        if (panel.Hover is null)
        {
            return;
        }

        foreach (var b in panel.Boxes)
        {
            if (Math.Abs(c.X - b.Position) <= b.Width / 2 && c.Y >= b.WhiskerLow && c.Y <= b.WhiskerHigh && b.Tooltip.Length > 0)
            {
                panel.Hover.LabelText = b.Tooltip;
                panel.Hover.Location = new Coordinates(c.X, c.Y);
                panel.Hover.IsVisible = true;
                panel.AvaPlot.Refresh();
                return;
            }
        }

        if (panel.Hover.IsVisible)
        {
            panel.Hover.IsVisible = false;
            panel.AvaPlot.Refresh();
        }
    }

    private void OnPointerExited(Panel panel)
    {
        CursorMoved?.Invoke(null);
        if (panel.Hover is { IsVisible: true })
        {
            panel.Hover.IsVisible = false;
            panel.AvaPlot.Refresh();
        }
    }

    private sealed class Panel
    {
        public Panel(AvaPlot avaPlot, int index)
        {
            AvaPlot = avaPlot;
            Index = index;
        }

        public AvaPlot AvaPlot { get; }
        public int Index { get; }
        public Text? Hover { get; set; }
        public List<BoxSpec> Boxes { get; } = new();
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
    }
}
