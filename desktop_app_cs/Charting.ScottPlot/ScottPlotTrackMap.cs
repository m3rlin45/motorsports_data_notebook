using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="TrackMapSpec"/>: coloured paths
/// and labelled points in equal units. Points that carry an id are
/// clickable: the pointer turns into a hand over them and a click reports
/// the id through <see cref="ChartPoint.ItemId"/>.
/// </summary>
public sealed class ScottPlotTrackMap : ScottPlotGridSurface<TrackMapSpec>
{
    private static readonly Avalonia.Input.Cursor HandCursor = new(StandardCursorType.Hand);
    private const double PickRadiusPx = 14;

    /// <inheritdoc />
    protected override void InstallInput(AvaPlot avaPlot, int index) => DataAreaInput.InstallFree(avaPlot.UserInputProcessor);

    /// <inheritdoc />
    protected override void Build(TrackMapSpec spec, TrackMapSpec? previous)
    {
        EnsurePlots(1, 1);
        var plot = PlotAt(0).Plot;
        MapStyle.Begin(plot, Theme, spec.Title);
        plot.Legend.IsVisible = spec.Paths.Any(p => p.Label.Length > 0);
        plot.Legend.Alignment = Alignment.UpperRight;

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

        MapStyle.Finish(plot, Theme);
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
