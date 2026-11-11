using System.Globalization;
using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="HistogramGridSpec"/>: one
/// <see cref="AvaPlot"/> per panel in a grid, shaded bands behind the bars,
/// side-by-side bars when a panel has several series, and a hover label
/// naming the bar under the pointer. The x cursor is reported like the strip
/// chart's; <see cref="SetCursor"/> is a no-op since bins, not samples, are
/// shown.
/// </summary>
public sealed class ScottPlotHistogramGrid : IChartSurface
{
    private const float LabelFontSize = 11;
    private ChartTheme _theme = ChartTheme.Dark;

    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();
    private HistogramGridSpec? _spec;

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
        if (spec is not HistogramGridSpec grid)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(_spec, grid))
        {
            return;
        }

        _spec = grid;
        EnsurePanelCount(grid.Panels.Count, grid.Columns);
        var rows = (grid.Panels.Count + grid.Columns - 1) / grid.Columns;

        for (var i = 0; i < grid.Panels.Count; i++)
        {
            var panelSpec = grid.Panels[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            var row = i / grid.Columns;
            var col = i % grid.Columns;

            plot.Clear();
            panel.Bars.Clear();
            ScottPlotStripChart.ApplyTheme(plot, _theme);
            plot.Title(panelSpec.Title, size: 12);
            plot.Axes.Bottom.Label.Text = row == rows - 1 ? grid.XLabel : string.Empty;
            plot.Axes.Left.Label.Text = col == 0 ? grid.YLabel : string.Empty;
            plot.Legend.IsVisible = i == 0 && panelSpec.Series.Count > 1;
            plot.Legend.Alignment = Alignment.UpperLeft;

            foreach (var band in grid.Bands)
            {
                var span = plot.Add.HorizontalSpan(band.Start, band.End);
                span.FillColor = Color.FromHex(band.Color).WithAlpha(band.Alpha);
                span.LineWidth = 0;
            }

            var n = panelSpec.Series.Count;
            var barWidth = panelSpec.BinWidth * (n == 1 ? 0.9 : 0.8 / n);
            for (var s = 0; s < n; s++)
            {
                var series = panelSpec.Series[s];
                var offset = n == 1 ? 0 : (s - (n - 1) / 2.0) * barWidth;
                var bars = new List<Bar>(series.Centers.Length);
                for (var b = 0; b < series.Centers.Length; b++)
                {
                    var colorHex = series.BarColors?[b] ?? series.Color ?? "#4682b4";
                    bars.Add(new Bar
                    {
                        Position = series.Centers[b] + offset,
                        Value = series.Values[b],
                        Size = barWidth,
                        FillColor = Color.FromHex(colorHex).WithAlpha(n == 1 ? 1 : 0.85),
                        LineWidth = 0,
                    });
                }

                var plottable = plot.Add.Bars(bars);
                plottable.LegendText = series.Label;
                panel.Bars.Add((series, offset, barWidth));
            }

            var zero = plot.Add.VerticalLine(0);
            zero.LineColor = Color.FromHex(_theme.Crosshair).WithAlpha(0.5);
            zero.LineWidth = 1;
            zero.LinePattern = LinePattern.Dashed;

            if (panelSpec.Annotation.Length > 0)
            {
                var note = plot.Add.Annotation(panelSpec.Annotation, Alignment.UpperRight);
                note.LabelFontColor = Color.FromHex(_theme.Text);
                note.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.85);
                note.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
                note.LabelFontSize = LabelFontSize;
            }

            panel.Hover = plot.Add.Text(string.Empty, 0, 0);
            panel.Hover.IsVisible = false;
            panel.Hover.LabelFontColor = Color.FromHex(_theme.Text);
            panel.Hover.LabelFontSize = LabelFontSize;
            panel.Hover.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.9);
            panel.Hover.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
            panel.Hover.LabelPadding = 3;
            panel.Hover.Alignment = Alignment.LowerLeft;
            panel.Hover.OffsetX = 10;
            panel.Hover.OffsetY = -10;
            panel.Title = panelSpec.Title;

            plot.Axes.AutoScale();
            plot.Axes.SetLimitsX(grid.XLimits.Min, grid.XLimits.Max);
            plot.Axes.Bottom.Range.Min = grid.XLimits.Min;
            plot.Axes.Left.Range.Min = 0;
            panel.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
        // Bins, not samples: nothing to mark. The hover label is driven by the pointer directly.
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
        _grid.ColumnDefinitions.Clear();
    }

    private void EnsurePanelCount(int count, int columns)
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
            _grid.Children.Add(avaPlot);
            _panels.Add(panel);
        }

        var rows = (count + columns - 1) / columns;
        _grid.RowDefinitions.Clear();
        _grid.ColumnDefinitions.Clear();
        for (var r = 0; r < rows; r++)
        {
            _grid.RowDefinitions.Add(new RowDefinition(1, GridUnitType.Star));
        }

        for (var c = 0; c < columns; c++)
        {
            _grid.ColumnDefinitions.Add(new ColumnDefinition(1, GridUnitType.Star));
        }

        for (var i = 0; i < _panels.Count; i++)
        {
            Grid.SetRow(_panels[i].AvaPlot, i / columns);
            Grid.SetColumn(_panels[i].AvaPlot, i % columns);
        }
    }

    private void Attach(Panel panel)
    {
        panel.OnPointerMoved = (_, e) => OnPointerMoved(panel, e);
        panel.OnPointerExited = (_, _) => OnPointerExited(panel);
        panel.OnPointerPressed = (_, e) => OnPointerPressed(panel, e);
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

        // The bar under the pointer: the series whose offset bin contains x, and whose height exceeds y.
        foreach (var (series, offset, width) in panel.Bars)
        {
            for (var b = 0; b < series.Centers.Length; b++)
            {
                var center = series.Centers[b] + offset;
                if (c.X >= center - width / 2 && c.X <= center + width / 2 && c.Y >= 0 && c.Y <= series.Values[b])
                {
                    var who = series.Label.Length > 0 ? $"{panel.Title} - {series.Label}" : panel.Title;
                    panel.Hover.LabelText = string.Format(CultureInfo.InvariantCulture, "{0}\nVelocity: {1:0} mm/s\nTime: {2:0.00}%", who, series.Centers[b], series.Values[b]);
                    panel.Hover.Location = new Coordinates(center, series.Values[b]);
                    panel.Hover.IsVisible = true;
                    panel.AvaPlot.Refresh();
                    return;
                }
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

    private void OnPointerPressed(Panel panel, PointerPressedEventArgs e)
    {
        var c = ToCoordinates(panel, e);
        Clicked?.Invoke(new ChartPoint(c.X, c.Y, panel.Index));
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
        public string Title { get; set; } = string.Empty;
        public Text? Hover { get; set; }
        public List<(BarSeriesSpec Series, double Offset, double Width)> Bars { get; } = new();
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
    }
}
