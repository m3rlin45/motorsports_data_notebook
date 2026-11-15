using Avalonia.Input;
using Avalonia.Threading;
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
/// <remarks>
/// The highlight is one preallocated plottable per (plot, series): a hover
/// writes the selected samples into its buffers and sets how many to draw,
/// so moving the pointer adds and removes nothing. Pointer moves are
/// coalesced to one highlight update per dispatcher pass.
/// </remarks>
public sealed class ScottPlotScatterGrid : ScottPlotGridSurface<ScatterGridSpec>
{
    private readonly List<Highlight> _highlights = new();
    private double _hoverRadius = 10;
    private double _snapRadius = 1e9;
    private bool _hasMap;
    private Marker? _snapMarker;
    private Coordinates? _pendingHover;
    private bool _hoverQueued;

    /// <summary>Number of samples currently highlighted (for tests).</summary>
    public int HighlightedCount { get; private set; }

    /// <inheritdoc />
    protected override void InstallInput(AvaPlot avaPlot, int index)
    {
        if (index == 0)
        {
            DataAreaInput.InstallFree(avaPlot.UserInputProcessor);
        }
        else
        {
            DataAreaInput.Install(avaPlot.UserInputProcessor);
        }
    }

    /// <inheritdoc />
    protected override void Build(ScatterGridSpec spec, ScatterGridSpec? previous)
    {
        EnsurePlots(spec.Panels.Count + 1, spec.Columns);
        _highlights.Clear();
        HighlightedCount = 0;

        BuildMap(PlotAt(0).Plot, spec);
        for (var i = 0; i < spec.Panels.Count; i++)
        {
            BuildPanel(PlotAt(i + 1).Plot, i + 1, spec.Panels[i]);
        }

        CommitView();
    }

    private void BuildMap(Plot plot, ScatterGridSpec spec)
    {
        MapStyle.Begin(plot, Theme, spec.MapTitle);
        plot.Legend.IsVisible = spec.Paths.Count > 1;
        plot.Legend.Alignment = Alignment.UpperRight;
        _hasMap = false;
        _snapMarker = null;

        if (spec.Paths.Count == 0)
        {
            PlotStyle.Caption(plot, Theme, "No GPS channels for track map", Alignment.MiddleCenter, Theme.Axis);
            plot.Axes.AutoScale();
            return;
        }

        double minX = double.PositiveInfinity, maxX = double.NegativeInfinity, minY = double.PositiveInfinity, maxY = double.NegativeInfinity;
        foreach (var path in spec.Paths)
        {
            var line = plot.Add.ScatterLine(path.X, path.Y);
            line.Color = ColorOf(path.Color).WithAlpha(PlotStyle.Alpha(Theme, 0.8));
            line.LineWidth = MapStyle.OutlineWidth * Theme.LineScale;
            line.LineStyle.Rounded = true;
            line.LegendText = path.Label;
            if (path.X.Length > 0)
            {
                minX = Math.Min(minX, path.X.Min());
                maxX = Math.Max(maxX, path.X.Max());
                minY = Math.Min(minY, path.Y.Min());
                maxY = Math.Max(maxY, path.Y.Max());
            }
        }

        // Highlights after every outline so they draw on top.
        foreach (var path in spec.Paths)
        {
            AddHighlight(plot, 0, path.SeriesId, path.X, path.Y, path.HighlightColor, size: 6);
        }

        _snapMarker = plot.Add.Marker(0, 0, MarkerShape.OpenCircle, 12, ColorOf(Theme.Crosshair));
        _snapMarker.IsVisible = false;
        _snapMarker.LineWidth = 1.5f;

        var extent = Math.Max(maxX - minX, maxY - minY);
        _hoverRadius = Math.Max(5.0, 0.015 * extent);
        _snapRadius = Math.Max(50.0, 0.15 * extent);
        MapStyle.Finish(plot, Theme);
        _hasMap = true;
    }

    private void BuildPanel(Plot plot, int plotIndex, ScatterPanelSpec panel)
    {
        plot.Clear();
        PlotStyle.Apply(plot, Theme);
        plot.Title(panel.Title, size: Theme.TitleSize);
        plot.Axes.Bottom.Label.Text = panel.XLabel;
        plot.Axes.Left.Label.Text = panel.YLabel;
        plot.Legend.IsVisible = panel.ShowLegend;
        plot.Legend.Alignment = Alignment.LowerRight;

        var zeroColor = ColorOf(Theme.Axis).WithAlpha(0.5);
        plot.Add.HorizontalLine(0, 1, zeroColor);
        plot.Add.VerticalLine(0, 1, zeroColor);

        foreach (var series in panel.Scatter)
        {
            var points = plot.Add.ScatterPoints(series.X, series.Y);
            points.Color = ColorOf(series.Color).WithAlpha(Theme.HighContrast ? Math.Max(series.Alpha, 0.5) : series.Alpha);
            points.MarkerSize = 2.5f;
            points.LegendText = series.Label;
        }

        foreach (var line in panel.Lines)
        {
            var fit = plot.Add.Line(line.X1, line.Y1, line.X2, line.Y2);
            fit.Color = ColorOf(line.Color);
            fit.LineWidth = 2 * Theme.LineScale;
        }

        foreach (var markers in panel.Markers)
        {
            var m = plot.Add.Markers(markers.X, markers.Y, MarkerShape.FilledCircle, 8, ColorOf(markers.Color));
            m.MarkerLineColor = ColorOf(Theme.DataBackground);
            m.MarkerLineWidth = 1;
        }

        foreach (var series in panel.Scatter)
        {
            AddHighlight(plot, plotIndex, series.SeriesId, series.X, series.Y, HighlightFor(series.SeriesId), size: 5);
        }

        if (panel.Annotation.Length > 0)
        {
            PlotStyle.Note(plot, Theme, panel.Annotation, Alignment.UpperLeft);
        }

        if (panel.Warning.Length > 0)
        {
            PlotStyle.Caption(plot, Theme, panel.Warning, Alignment.LowerRight, ChartColors.Warning);
        }

        plot.Axes.AutoScale();
    }

    private void AddHighlight(Plot plot, int plotIndex, int seriesId, double[] xs, double[] ys, string color, float size)
    {
        var n = Math.Min(xs.Length, ys.Length);
        if (n == 0)
        {
            return;
        }

        var hx = new double[n];
        var hy = new double[n];
        var points = plot.Add.ScatterPoints(hx, hy);
        points.Color = ColorOf(color);
        points.MarkerSize = size;
        points.MarkerLineColor = ColorOf(Theme.DataBackground);
        points.MarkerLineWidth = 0.5f;
        points.IsVisible = false;
        _highlights.Add(new Highlight(plotIndex, seriesId, xs, ys, hx, hy, points));
    }

    private string HighlightFor(int seriesId) =>
        Spec?.Paths.FirstOrDefault(p => p.SeriesId == seriesId)?.HighlightColor ?? (seriesId == 0 ? ChartColors.HighlightA : ChartColors.HighlightB);

    /// <summary>Highlight the samples within the hover radius of map point (<paramref name="sx"/>, <paramref name="sy"/>).</summary>
    public void HighlightNear(double sx, double sy)
    {
        var count = 0;
        var maskBySeries = new Dictionary<int, bool[]>();
        var r2 = _hoverRadius * _hoverRadius;
        foreach (var path in Spec?.Paths ?? Array.Empty<TrackPathSpec>())
        {
            var mask = new bool[path.X.Length];
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

        foreach (var hl in _highlights)
        {
            var n = 0;
            if (maskBySeries.TryGetValue(hl.SeriesId, out var mask))
            {
                var limit = Math.Min(mask.Length, hl.X.Length);
                for (var i = 0; i < limit; i++)
                {
                    if (mask[i])
                    {
                        hl.BufferX[n] = hl.X[i];
                        hl.BufferY[n] = hl.Y[i];
                        n++;
                    }
                }
            }

            hl.Points.IsVisible = n > 0;
            hl.Points.MinRenderIndex = 0;
            hl.Points.MaxRenderIndex = Math.Max(n - 1, 0);
        }

        if (_snapMarker is not null)
        {
            _snapMarker.Location = new Coordinates(sx, sy);
            _snapMarker.IsVisible = true;
        }

        HighlightedCount = count;
        RefreshAll();
    }

    /// <summary>Remove the highlight.</summary>
    public void ClearHighlights()
    {
        var had = HighlightedCount > 0 || _snapMarker is { IsVisible: true };
        foreach (var hl in _highlights)
        {
            hl.Points.IsVisible = false;
        }

        if (_snapMarker is not null)
        {
            _snapMarker.IsVisible = false;
        }

        HighlightedCount = 0;
        if (had)
        {
            RefreshAll();
        }
    }

    private void RefreshAll()
    {
        foreach (var p in PlotControls)
        {
            p.Refresh();
        }
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(int index, Coordinates at, PointerEventArgs e)
    {
        if (index != 0)
        {
            return; // positions on the scatter panels are 2-D; only the map drives the cursor and the highlight
        }

        RaiseCursorMoved(at.X);
        if (!_hasMap)
        {
            return;
        }

        // Several pointer events can arrive per frame; act on the latest one only.
        _pendingHover = at;
        if (!_hoverQueued)
        {
            _hoverQueued = true;
            Dispatcher.UIThread.Post(ApplyPendingHover, DispatcherPriority.Input);
        }
    }

    private void ApplyPendingHover()
    {
        _hoverQueued = false;
        if (_pendingHover is not { } at || Spec is null || !_hasMap)
        {
            return;
        }

        _pendingHover = null;

        // Snap to the nearest track point within the snap radius.
        var bestD2 = _snapRadius * _snapRadius;
        (double X, double Y)? best = null;
        foreach (var path in Spec.Paths)
        {
            for (var i = 0; i < path.X.Length; i++)
            {
                var dx = path.X[i] - at.X;
                var dy = path.Y[i] - at.Y;
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

    /// <inheritdoc />
    protected override void OnPointerExited(int index)
    {
        if (index != 0)
        {
            return;
        }

        _pendingHover = null;
        RaiseCursorMoved(null);
        ClearHighlights();
    }

    /// <summary>One series' highlight on one plot: the source samples, the buffers drawn, and the plottable.</summary>
    private sealed record Highlight(int PlotIndex, int SeriesId, double[] X, double[] Y, double[] BufferX, double[] BufferY, Scatter Points);
}
