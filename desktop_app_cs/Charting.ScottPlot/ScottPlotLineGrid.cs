using System.Globalization;
using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>ScottPlot 5 implementation of a <see cref="LineGridSpec"/>: line-and-marker series per panel with a hover label naming the nearest point.</summary>
public sealed class ScottPlotLineGrid : IChartSurface
{
    private readonly Grid _grid = new();
    private readonly List<Panel> _panels = new();
    private LineGridSpec? _spec;
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
        if (spec is not LineGridSpec grid)
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
        var text = Color.FromHex(_theme.Text);

        for (var i = 0; i < grid.Panels.Count; i++)
        {
            var panelSpec = grid.Panels[i];
            var panel = _panels[i];
            var plot = panel.AvaPlot.Plot;
            plot.Clear();
            panel.Series.Clear();
            ScottPlotStripChart.ApplyTheme(plot, _theme);
            plot.Title(panelSpec.Title, size: 12);
            plot.Axes.Bottom.Label.Text = i / grid.Columns == rows - 1 ? grid.XLabel : string.Empty;
            plot.Axes.Left.Label.Text = i % grid.Columns == 0 ? grid.YLabel : string.Empty;
            plot.Legend.IsVisible = i == 0 && panelSpec.Series.Count > 1;
            plot.Legend.Alignment = Alignment.LowerRight;

            foreach (var series in panelSpec.Series)
            {
                if (series.X.Length == 0)
                {
                    continue;
                }

                var line = plot.Add.Scatter(series.X, series.Y);
                line.Color = Color.FromHex(series.Color);
                line.LineWidth = 2;
                line.MarkerSize = 7;
                line.LegendText = series.Label;
                panel.Series.Add(series);
            }

            if (panelSpec.Annotation.Length > 0)
            {
                var note = plot.Add.Annotation(panelSpec.Annotation, Alignment.UpperRight);
                note.LabelFontColor = text;
                note.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.85);
                note.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
                note.LabelFontSize = 11;
            }

            panel.Hover = plot.Add.Text(string.Empty, 0, 0);
            panel.Hover.IsVisible = false;
            panel.Hover.LabelFontColor = text;
            panel.Hover.LabelFontSize = 11;
            panel.Hover.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.9);
            panel.Hover.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
            panel.Hover.LabelPadding = 3;
            panel.Hover.Alignment = Alignment.LowerLeft;
            panel.Hover.OffsetX = 10;
            panel.Hover.OffsetY = -10;
            panel.Title = panelSpec.Title;

            plot.Axes.AutoScale();
            panel.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var panel in _panels)
        {
            panel.AvaPlot.PointerMoved -= panel.OnPointerMoved;
            panel.AvaPlot.PointerExited -= panel.OnPointerExited;
            panel.AvaPlot.PointerPressed -= panel.OnPointerPressed;
        }

        _panels.Clear();
        _grid.Children.Clear();
    }

    private void EnsurePanelCount(int count, int columns)
    {
        while (_panels.Count > count)
        {
            var last = _panels[^1];
            last.AvaPlot.PointerMoved -= last.OnPointerMoved;
            last.AvaPlot.PointerExited -= last.OnPointerExited;
            last.AvaPlot.PointerPressed -= last.OnPointerPressed;
            _grid.Children.Remove(last.AvaPlot);
            _panels.RemoveAt(_panels.Count - 1);
        }

        while (_panels.Count < count)
        {
            var avaPlot = new AvaPlot();
            var panel = new Panel(avaPlot, _panels.Count);
            panel.OnPointerMoved = (_, e) => OnPointerMoved(panel, e);
            panel.OnPointerExited = (_, _) => OnPointerExited(panel);
            panel.OnPointerPressed = (_, e) =>
            {
                var c = ToCoordinates(panel, e);
                Clicked?.Invoke(new ChartPoint(c.X, c.Y, panel.Index));
            };
            avaPlot.PointerMoved += panel.OnPointerMoved;
            avaPlot.PointerExited += panel.OnPointerExited;
            avaPlot.PointerPressed += panel.OnPointerPressed;
            DataAreaInput.Install(avaPlot.UserInputProcessor);
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

        // Nearest point within ~8 px, measured in pixels so both axes count equally.
        var plot = panel.AvaPlot.Plot;
        var pointer = plot.GetPixel(c);
        var bestD2 = 8.0 * 8.0;
        (LineSeriesSpec Series, int Index)? best = null;
        foreach (var series in panel.Series)
        {
            for (var i = 0; i < series.X.Length; i++)
            {
                var px = plot.GetPixel(new Coordinates(series.X[i], series.Y[i]));
                var dx = px.X - pointer.X;
                var dy = px.Y - pointer.Y;
                var d2 = dx * dx + dy * dy;
                if (d2 <= bestD2)
                {
                    bestD2 = d2;
                    best = (series, i);
                }
            }
        }

        if (best is { } hit)
        {
            var who = hit.Series.Label.Length > 0 ? $"{panel.Title} - {hit.Series.Label}" : panel.Title;
            var count = hit.Index < hit.Series.Counts.Length ? hit.Series.Counts[hit.Index] : 0;
            panel.Hover.LabelText = string.Format(CultureInfo.InvariantCulture, "{0}\nMetric: {1:0.0}\nG: {2:0.00}\nn = {3}", who, hit.Series.X[hit.Index], hit.Series.Y[hit.Index], count);
            panel.Hover.Location = new Coordinates(hit.Series.X[hit.Index], hit.Series.Y[hit.Index]);
            panel.Hover.IsVisible = true;
            panel.AvaPlot.Refresh();
        }
        else if (panel.Hover.IsVisible)
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
        public string Title { get; set; } = string.Empty;
        public Text? Hover { get; set; }
        public List<LineSeriesSpec> Series { get; } = new();
        public EventHandler<PointerEventArgs> OnPointerMoved { get; set; } = (_, _) => { };
        public EventHandler<PointerEventArgs> OnPointerExited { get; set; } = (_, _) => { };
        public EventHandler<PointerPressedEventArgs> OnPointerPressed { get; set; } = (_, _) => { };
    }
}
