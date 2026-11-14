using System.Globalization;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>ScottPlot 5 implementation of a <see cref="LineGridSpec"/>: line-and-marker series per panel with a hover label naming the nearest point.</summary>
public sealed class ScottPlotLineGrid : ScottPlotGridSurface<LineGridSpec>
{
    private const double HoverRadiusPx = 8;
    private readonly List<Panel> _panels = new();

    /// <inheritdoc />
    protected override void OnPlotCreated(AvaPlot avaPlot, int index) => _panels.Add(new Panel());

    /// <inheritdoc />
    protected override void Build(LineGridSpec spec, LineGridSpec? previous)
    {
        EnsurePlots(spec.Panels.Count, spec.Columns);
        _panels.RemoveRange(PlotCount, _panels.Count - PlotCount);
        var rows = (spec.Panels.Count + spec.Columns - 1) / spec.Columns;

        for (var i = 0; i < spec.Panels.Count; i++)
        {
            var panelSpec = spec.Panels[i];
            var panel = _panels[i];
            var plot = PlotAt(i).Plot;
            plot.Clear();
            panel.Series.Clear();
            PlotStyle.Apply(plot, Theme);
            plot.Title(panelSpec.Title, size: Theme.TitleSize);
            plot.Axes.Bottom.Label.Text = i / spec.Columns == rows - 1 ? spec.XLabel : string.Empty;
            plot.Axes.Left.Label.Text = i % spec.Columns == 0 ? spec.YLabel : string.Empty;
            plot.Legend.IsVisible = i == 0 && panelSpec.Series.Count > 1;
            plot.Legend.Alignment = Alignment.LowerRight;

            foreach (var series in panelSpec.Series)
            {
                if (series.X.Length == 0)
                {
                    continue;
                }

                var line = plot.Add.Scatter(series.X, series.Y);
                line.Color = ColorOf(series.Color);
                line.LineWidth = 2 * Theme.LineScale;
                line.MarkerSize = 8;
                line.LegendText = series.Label;
                panel.Series.Add(series);
            }

            if (panelSpec.Annotation.Length > 0)
            {
                PlotStyle.Note(plot, Theme, panelSpec.Annotation, Alignment.UpperRight);
            }

            panel.Hover = PlotStyle.HoverLabel(plot, Theme);
            panel.Title = panelSpec.Title;
            plot.Axes.AutoScale();
        }

        CommitView();
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(int index, Coordinates at, PointerEventArgs e)
    {
        base.OnPointerMoved(index, at, e);
        var panel = _panels[index];
        if (panel.Hover is null || Spec is null)
        {
            return;
        }

        // Nearest point within a few pixels, measured in pixels so both axes count equally.
        var plot = PlotAt(index).Plot;
        var pointer = plot.GetPixel(at);
        var bestD2 = HoverRadiusPx * HoverRadiusPx;
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
            var text = string.Format(CultureInfo.InvariantCulture, Spec.HoverFormat, who, hit.Series.X[hit.Index], hit.Series.Y[hit.Index], count);
            PlotStyle.ShowHover(panel.Hover, text, new Coordinates(hit.Series.X[hit.Index], hit.Series.Y[hit.Index]));
            PlotAt(index).Refresh();
        }
        else
        {
            HideHover(index);
        }
    }

    /// <inheritdoc />
    protected override void OnPointerExited(int index)
    {
        base.OnPointerExited(index);
        HideHover(index);
    }

    private void HideHover(int index)
    {
        if (_panels[index].Hover is { IsVisible: true } hover)
        {
            hover.IsVisible = false;
            PlotAt(index).Refresh();
        }
    }

    private sealed class Panel
    {
        public string Title { get; set; } = string.Empty;
        public Text? Hover { get; set; }
        public List<LineSeriesSpec> Series { get; } = new();
    }
}
