using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="ScatterGridSpec"/>: the track
/// map in the first cell, scatter panels after it. Pointer motion over the
/// map snaps to the nearest track point and highlights the samples within a
/// radius of it on the map and on every panel (index-aligned per series).
/// </summary>
public sealed class ScottPlotScatterGrid : IChartSurface
{
    private const float LabelFontSize = 11;
    private const string MapTitle = "Track map (hover near the line to highlight samples)";

    private readonly Grid _grid = new();
    private readonly List<AvaPlot> _plots = new();
    private readonly List<Scatter> _highlights = new(); // one per (panel, series), re-created per hover
    private readonly List<(int Panel, int SeriesId, double[] X, double[] Y, string Color)> _highlightSources = new();
    private ScatterGridSpec? _spec;
    private ChartTheme _theme = ChartTheme.Dark;
    private AvaPlot? _map;
    private double _hoverRadius = 10;
    private double _snapRadius = 1e9;
    private EventHandler<PointerEventArgs>? _mapMoved;
    private EventHandler<PointerEventArgs>? _mapExited;
    private Marker? _snapMarker;

    /// <inheritdoc />
    public Control Control => _grid;

    /// <inheritdoc />
    public event Action<double?>? CursorMoved;

    /// <inheritdoc />
    public event Action<ChartPoint>? Clicked;

    /// <summary>Number of samples currently highlighted (for tests).</summary>
    public int HighlightedCount { get; private set; }

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
        if (spec is not ScatterGridSpec grid)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(_spec, grid))
        {
            return;
        }

        _spec = grid;
        EnsurePlotCount(grid.Panels.Count + 1, grid.Columns);
        _highlights.Clear();
        _highlightSources.Clear();
        HighlightedCount = 0;

        RenderMap(_plots[0], grid);
        for (var i = 0; i < grid.Panels.Count; i++)
        {
            RenderPanel(_plots[i + 1], i + 1, grid.Panels[i]);
        }
    }

    private void RenderMap(AvaPlot avaPlot, ScatterGridSpec grid)
    {
        var plot = avaPlot.Plot;
        plot.Clear();
        ScottPlotStripChart.ApplyTheme(plot, _theme);
        plot.Title(MapTitle, size: 11);
        plot.Legend.IsVisible = grid.Paths.Count > 1;
        plot.Legend.Alignment = Alignment.UpperRight;
        _map = null;
        _snapMarker = null;

        if (grid.Paths.Count == 0)
        {
            var note = plot.Add.Annotation("No GPS channels for track map", Alignment.MiddleCenter);
            note.LabelFontColor = Color.FromHex(_theme.Text);
            note.LabelBackgroundColor = Colors.Transparent;
            note.LabelBorderColor = Colors.Transparent;
            plot.Axes.AutoScale();
            avaPlot.Refresh();
            return;
        }

        double minX = double.PositiveInfinity, maxX = double.NegativeInfinity, minY = double.PositiveInfinity, maxY = double.NegativeInfinity;
        foreach (var path in grid.Paths)
        {
            var line = plot.Add.ScatterLine(path.X, path.Y);
            line.Color = Color.FromHex(path.Color).WithAlpha(0.7);
            line.LineWidth = 1;
            line.LegendText = path.Label;
            _highlightSources.Add((0, path.SeriesId, path.X, path.Y, path.HighlightColor));
            if (path.X.Length > 0)
            {
                minX = Math.Min(minX, path.X.Min());
                maxX = Math.Max(maxX, path.X.Max());
                minY = Math.Min(minY, path.Y.Min());
                maxY = Math.Max(maxY, path.Y.Max());
            }
        }

        _snapMarker = plot.Add.Marker(0, 0, MarkerShape.OpenCircle, 12, Color.FromHex(_theme.Crosshair));
        _snapMarker.IsVisible = false;
        _snapMarker.LineWidth = 1.5f;

        var extent = Math.Max(maxX - minX, maxY - minY);
        _hoverRadius = Math.Max(5.0, 0.015 * extent);
        _snapRadius = Math.Max(50.0, 0.15 * extent);
        plot.Axes.Bottom.Label.Text = "x (m)";
        plot.Axes.Left.Label.Text = "y (m)";
        plot.Axes.AutoScale();
        // Equal metres per pixel; zoom the shorter axis out rather than cropping the longer one.
        plot.Axes.Rules.Add(new global::ScottPlot.AxisRules.SquareZoomOut(plot.Axes.Bottom, plot.Axes.Left));
        _map = avaPlot;
        avaPlot.Refresh();
    }

    private void RenderPanel(AvaPlot avaPlot, int panelIndex, ScatterPanelSpec panel)
    {
        var plot = avaPlot.Plot;
        plot.Clear();
        ScottPlotStripChart.ApplyTheme(plot, _theme);
        plot.Title(panel.Title, size: 11);
        plot.Axes.Bottom.Label.Text = panel.XLabel;
        plot.Axes.Left.Label.Text = panel.YLabel;
        plot.Legend.IsVisible = panel.ShowLegend;
        plot.Legend.Alignment = Alignment.LowerRight;

        var zeroColor = Color.FromHex(_theme.Grid);
        plot.Add.HorizontalLine(0, 1, zeroColor);
        plot.Add.VerticalLine(0, 1, zeroColor);

        foreach (var series in panel.Scatter)
        {
            var points = plot.Add.ScatterPoints(series.X, series.Y);
            points.Color = Color.FromHex(series.Color).WithAlpha(series.Alpha);
            points.MarkerSize = 2.5f;
            points.LegendText = series.Label;
            _highlightSources.Add((panelIndex, series.SeriesId, series.X, series.Y, HighlightFor(series.SeriesId)));
        }

        foreach (var line in panel.Lines)
        {
            var fit = plot.Add.Line(line.X1, line.Y1, line.X2, line.Y2);
            fit.Color = Color.FromHex(line.Color);
            fit.LineWidth = 2;
        }

        foreach (var markers in panel.Markers)
        {
            var m = plot.Add.Markers(markers.X, markers.Y, MarkerShape.FilledCircle, 6, Color.FromHex(markers.Color));
            m.MarkerLineColor = Color.FromHex(_theme.Text);
            m.MarkerLineWidth = 0.6f;
        }

        if (panel.Annotation.Length > 0)
        {
            var note = plot.Add.Annotation(panel.Annotation, Alignment.UpperLeft);
            note.LabelFontColor = Color.FromHex(_theme.Text);
            note.LabelBackgroundColor = Color.FromHex(_theme.PanelBackground).WithAlpha(0.85);
            note.LabelBorderColor = Color.FromHex(_theme.PanelOutline);
            note.LabelFontSize = LabelFontSize;
        }

        if (panel.Warning.Length > 0)
        {
            var warning = plot.Add.Annotation(panel.Warning, Alignment.LowerRight);
            warning.LabelFontColor = Color.FromHex("#ff8080");
            warning.LabelBackgroundColor = Colors.Transparent;
            warning.LabelBorderColor = Colors.Transparent;
            warning.LabelFontSize = LabelFontSize - 1;
        }

        plot.Axes.AutoScale();
        avaPlot.Refresh();
    }

    private string HighlightFor(int seriesId) =>
        _spec?.Paths.FirstOrDefault(p => p.SeriesId == seriesId)?.HighlightColor ?? (seriesId == 0 ? "#ffe066" : "#ff6ec7");

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
        // Positions here are 2-D; the track-map hover drives the highlight instead.
    }

    /// <summary>Highlight the samples within the hover radius of map point (<paramref name="sx"/>, <paramref name="sy"/>).</summary>
    public void HighlightNear(double sx, double sy)
    {
        ClearHighlights(refresh: false);
        var count = 0;
        var maskBySeries = new Dictionary<int, bool[]>();
        foreach (var path in _spec?.Paths ?? Array.Empty<TrackPathSpec>())
        {
            var mask = new bool[path.X.Length];
            var r2 = _hoverRadius * _hoverRadius;
            for (var i = 0; i < mask.Length; i++)
            {
                var dx = path.X[i] - sx;
                var dy = path.Y[i] - sy;
                mask[i] = dx * dx + dy * dy <= r2;
                if (mask[i])
                {
                    count++;
                }
            }

            maskBySeries[path.SeriesId] = mask;
        }

        foreach (var (panelIndex, seriesId, xs, ys, color) in _highlightSources)
        {
            if (!maskBySeries.TryGetValue(seriesId, out var mask))
            {
                continue;
            }

            var hx = new List<double>();
            var hy = new List<double>();
            for (var i = 0; i < Math.Min(mask.Length, xs.Length); i++)
            {
                if (mask[i])
                {
                    hx.Add(xs[i]);
                    hy.Add(ys[i]);
                }
            }

            if (hx.Count == 0)
            {
                continue;
            }

            var plot = _plots[panelIndex].Plot;
            var hl = plot.Add.ScatterPoints(hx.ToArray(), hy.ToArray());
            hl.Color = Color.FromHex(color);
            hl.MarkerSize = panelIndex == 0 ? 6 : 5;
            hl.MarkerLineColor = Colors.Black;
            hl.MarkerLineWidth = 0.4f;
            _highlights.Add(hl);
        }

        if (_snapMarker is not null)
        {
            _snapMarker.Location = new Coordinates(sx, sy);
            _snapMarker.IsVisible = true;
        }

        HighlightedCount = count;
        foreach (var p in _plots)
        {
            p.Refresh();
        }
    }

    /// <summary>Remove the highlight.</summary>
    public void ClearHighlights() => ClearHighlights(refresh: true);

    private void ClearHighlights(bool refresh)
    {
        foreach (var hl in _highlights)
        {
            foreach (var p in _plots)
            {
                p.Plot.Remove(hl);
            }
        }

        _highlights.Clear();
        if (_snapMarker is not null)
        {
            _snapMarker.IsVisible = false;
        }

        var had = HighlightedCount > 0;
        HighlightedCount = 0;
        if (refresh && had)
        {
            foreach (var p in _plots)
            {
                p.Refresh();
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        DetachMap();
        _plots.Clear();
        _grid.Children.Clear();
        _grid.RowDefinitions.Clear();
        _grid.ColumnDefinitions.Clear();
    }

    private void EnsurePlotCount(int count, int columns)
    {
        DetachMap();
        while (_plots.Count > count)
        {
            _grid.Children.Remove(_plots[^1]);
            _plots.RemoveAt(_plots.Count - 1);
        }

        while (_plots.Count < count)
        {
            var avaPlot = new AvaPlot();
            DataAreaInput.Install(avaPlot.UserInputProcessor);
            var index = _plots.Count;
            avaPlot.PointerPressed += (_, e) =>
            {
                var c = ToCoordinates(avaPlot, e);
                Clicked?.Invoke(new ChartPoint(c.X, c.Y, index));
            };
            _grid.Children.Add(avaPlot);
            _plots.Add(avaPlot);
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

        for (var i = 0; i < _plots.Count; i++)
        {
            Grid.SetRow(_plots[i], i / columns);
            Grid.SetColumn(_plots[i], i % columns);
        }

        // The first plot is the map: its pointer drives the highlight.
        _mapMoved = (_, e) => OnMapPointerMoved(e);
        _mapExited = (_, _) =>
        {
            CursorMoved?.Invoke(null);
            ClearHighlights();
        };
        _plots[0].PointerMoved += _mapMoved;
        _plots[0].PointerExited += _mapExited;
    }

    private void DetachMap()
    {
        if (_plots.Count > 0 && _mapMoved is not null)
        {
            _plots[0].PointerMoved -= _mapMoved;
            _plots[0].PointerExited -= _mapExited;
        }

        _mapMoved = null;
        _mapExited = null;
    }

    private static Coordinates ToCoordinates(AvaPlot avaPlot, PointerEventArgs e)
    {
        var position = e.GetPosition(avaPlot);
        var scale = avaPlot.DisplayScale;
        return avaPlot.Plot.GetCoordinates(new Pixel((float)(position.X * scale), (float)(position.Y * scale)));
    }

    private void OnMapPointerMoved(PointerEventArgs e)
    {
        if (_map is null || _spec is null)
        {
            return;
        }

        var c = ToCoordinates(_map, e);
        CursorMoved?.Invoke(c.X);

        // Snap to the nearest track point within the snap radius.
        var bestD2 = _snapRadius * _snapRadius;
        (double X, double Y)? best = null;
        foreach (var path in _spec.Paths)
        {
            for (var i = 0; i < path.X.Length; i++)
            {
                var dx = path.X[i] - c.X;
                var dy = path.Y[i] - c.Y;
                var d2 = dx * dx + dy * dy;
                if (d2 <= bestD2)
                {
                    bestD2 = d2;
                    best = (path.X[i], path.Y[i]);
                }
            }
        }

        if (best is { } snap)
        {
            HighlightNear(snap.X, snap.Y);
        }
        else
        {
            ClearHighlights();
        }
    }
}
