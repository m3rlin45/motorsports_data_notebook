using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="TrackMapSpec"/>: coloured paths
/// and labelled points in equal units. Points that carry an id are
/// clickable: the pointer turns into a hand over them and a click reports
/// the id through <see cref="ChartPoint.ItemId"/>. The spec's highlights are
/// built once and only shown or hidden by <see cref="HighlightItem"/>, so
/// following the pointer on another chart costs a redraw, not a rebuild.
/// The same goes for the position marker of a spec with a
/// <see cref="TrackMapSpec.Cursor"/>: <see cref="SetCursor"/> moves one
/// marker along that line.
/// </summary>
public sealed class ScottPlotTrackMap : ScottPlotGridSurface<TrackMapSpec>
{
    private static readonly Avalonia.Input.Cursor HandCursor = new(StandardCursorType.Hand);
    private const double PickRadiusPx = 14;
    private readonly List<(int Id, IPlottable Plottable)> _highlights = new();
    private int _highlighted = -1;
    private global::ScottPlot.Plottables.Marker? _cursorMarker;
    private double? _cursorAt;

    /// <summary>Where the position marker is drawn (map units), or null while it is hidden.</summary>
    internal (double X, double Y)? CursorPoint => _cursorMarker is { IsVisible: true } m ? (m.X, m.Y) : null;

    /// <inheritdoc />
    public override void SetCursor(double? x)
    {
        if (x == _cursorAt)
        {
            return;
        }

        _cursorAt = x;
        PlaceCursor();
        if (Spec is not null)
        {
            PlotAt(0).Refresh();
        }
    }

    /// <summary>Put the marker where <see cref="_cursorAt"/> falls on the spec's cursor line, or hide it.</summary>
    private void PlaceCursor()
    {
        if (_cursorMarker is null)
        {
            return;
        }

        if (_cursorAt is not { } at || Spec?.Cursor is not { Along.Length: > 0 } line || double.IsNaN(at))
        {
            _cursorMarker.IsVisible = false;
            return;
        }

        // The first sample at or past the position; the point between it and the one before, in proportion.
        var next = Array.BinarySearch(line.Along, at);
        if (next < 0)
        {
            next = ~next;
        }

        double px, py;
        if (next <= 0)
        {
            (px, py) = (line.X[0], line.Y[0]);
        }
        else if (next >= line.Along.Length)
        {
            (px, py) = (line.X[^1], line.Y[^1]);
        }
        else
        {
            var span = line.Along[next] - line.Along[next - 1];
            var t = span > 0 ? (at - line.Along[next - 1]) / span : 0;
            px = line.X[next - 1] + t * (line.X[next] - line.X[next - 1]);
            py = line.Y[next - 1] + t * (line.Y[next] - line.Y[next - 1]);
        }

        _cursorMarker.X = px;
        _cursorMarker.Y = py;
        _cursorMarker.IsVisible = true;
    }

    /// <summary>The id whose highlight is showing, or -1.</summary>
    internal int HighlightedItem => _highlighted;

    /// <summary>How many plottables are showing for the current highlight.</summary>
    internal int VisibleHighlightParts => _highlights.Count(h => h.Plottable.IsVisible);

    /// <inheritdoc />
    public override void HighlightItem(int id)
    {
        if (id == _highlighted)
        {
            return;
        }

        _highlighted = id;
        ShowHighlight();
        if (Spec is not null)
        {
            PlotAt(0).Refresh();
        }
    }

    private void ShowHighlight()
    {
        foreach (var (id, plottable) in _highlights)
        {
            plottable.IsVisible = id == _highlighted;
        }
    }

    /// <inheritdoc />
    protected override void InstallInput(AvaPlot avaPlot, int index) => DataAreaInput.InstallFree(avaPlot.UserInputProcessor);

    /// <inheritdoc />
    protected override void Build(TrackMapSpec spec, TrackMapSpec? previous)
    {
        EnsurePlots(1, 1);
        var plot = PlotAt(0).Plot;
        MapStyle.Begin(plot, Theme, spec.Title);
        plot.Legend.IsVisible = !spec.Compact && spec.Paths.Any(p => p.Label.Length > 0);
        plot.Legend.Alignment = Alignment.UpperRight;

        // A companion map stays put (it is too small to explore and is there to show where something
        // is) and sits on its card without a panel of its own.
        PlotAt(0).UserInputProcessor.IsEnabled = !spec.Compact;
        if (spec.Compact)
        {
            plot.DataBackground.Color = plot.FigureBackground.Color;
        }

        foreach (var path in spec.Paths)
        {
            var line = plot.Add.ScatterLine(path.X, path.Y);
            line.Color = ColorOf(path.Color);
            line.LineWidth = (float)path.Width * Theme.LineScale;
            line.LineStyle.Rounded = true;
            line.LegendText = path.Label;
        }

        foreach (var point in spec.Points)
        {
            var marker = plot.Add.Marker(point.X, point.Y, point.Star ? MarkerShape.FilledDiamond : MarkerShape.FilledCircle, (float)point.Size, ColorOf(point.Color));
            marker.MarkerLineColor = ColorOf(Theme.DataBackground);
            marker.MarkerLineWidth = 1;
            if (point.Label.Length > 0)
            {
                var text = plot.Add.Text(point.Label, point.X, point.Y);
                text.LabelFontColor = ColorOf(Theme.Text);
                text.LabelFontSize = Theme.TickSize;
                text.LabelFontName = Fonts.Detect(point.Label);
                text.OffsetX = 8;
                text.OffsetY = -8;
            }
        }

        _highlights.Clear();
        foreach (var highlight in spec.Highlights ?? Array.Empty<TrackMapHighlightSpec>())
        {
            var line = plot.Add.ScatterLine(highlight.X, highlight.Y);
            line.Color = ColorOf(highlight.Color);
            line.LineWidth = (float)highlight.Width * Theme.LineScale;
            line.LineStyle.Rounded = true;
            _highlights.Add((highlight.Id, line));

            var marker = plot.Add.Marker(highlight.MarkerX, highlight.MarkerY, MarkerShape.FilledCircle, (float)highlight.Width + 5, ColorOf(highlight.Color));
            marker.MarkerLineColor = ColorOf(Theme.DataBackground);
            marker.MarkerLineWidth = 1.5f;
            _highlights.Add((highlight.Id, marker));

            if (highlight.Label.Length > 0)
            {
                var text = plot.Add.Text(highlight.Label, highlight.MarkerX, highlight.MarkerY);
                text.LabelFontColor = ColorOf(Theme.Text);
                text.LabelFontSize = Theme.TickSize;
                text.LabelFontName = Fonts.Detect(highlight.Label);
                text.LabelBold = true;
                text.OffsetX = 8;
                text.OffsetY = -8;
                _highlights.Add((highlight.Id, text));
            }
        }

        // Last, so it is on top of everything: the position marker, hidden until SetCursor places it.
        _cursorMarker = null;
        if (spec.Cursor is { } cursor)
        {
            _cursorMarker = plot.Add.Marker(0, 0, MarkerShape.FilledCircle, (float)cursor.Size, ColorOf(cursor.Color));
            _cursorMarker.MarkerLineColor = ColorOf(Theme.DataBackground);
            _cursorMarker.MarkerLineWidth = 2;
            PlaceCursor();
        }

        ShowHighlight();
        MapStyle.Finish(plot, Theme, scaleBar: !spec.Compact);
        CommitView();
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(int index, Coordinates at, PointerEventArgs e)
    {
        base.OnPointerMoved(index, at, e);
        PlotAt(0).Cursor = PointNear(at) >= 0 ? HandCursor : Avalonia.Input.Cursor.Default;
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(int index, Coordinates at, PointerPressedEventArgs e) =>
        RaiseClicked(new ChartPoint(at.X, at.Y, index, PointNear(at)));

    /// <summary>Id of the clickable point within a few pixels of <paramref name="at"/>, or -1.</summary>
    internal int PointNear(Coordinates at)
    {
        if (Spec is null)
        {
            return -1;
        }

        var plot = PlotAt(0).Plot;
        var pointer = plot.GetPixel(at);
        var best = -1;
        var bestD2 = PickRadiusPx * PickRadiusPx;
        foreach (var point in Spec.Points)
        {
            if (point.Id < 0)
            {
                continue;
            }

            var px = plot.GetPixel(new Coordinates(point.X, point.Y));
            var dx = px.X - pointer.X;
            var dy = px.Y - pointer.Y;
            var d2 = dx * dx + dy * dy;
            if (d2 <= bestD2)
            {
                bestD2 = d2;
                best = point.Id;
            }
        }

        return best;
    }
}
