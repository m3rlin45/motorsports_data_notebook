using System.Globalization;
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
/// chart's; <see cref="IChartSurface.SetCursor"/> is a no-op since bins, not
/// samples, are shown.
/// </summary>
public sealed class ScottPlotHistogramGrid : ScottPlotGridSurface<HistogramGridSpec>
{
    private readonly List<Panel> _panels = new();

    /// <inheritdoc />
    protected override void OnPlotCreated(AvaPlot avaPlot, int index) => _panels.Add(new Panel());

    /// <inheritdoc />
    protected override void Build(HistogramGridSpec spec, HistogramGridSpec? previous)
    {
        EnsurePlots(spec.Panels.Count, spec.Columns);
        _panels.RemoveRange(PlotCount, _panels.Count - PlotCount);
        var rows = (spec.Panels.Count + spec.Columns - 1) / spec.Columns;

        for (var i = 0; i < spec.Panels.Count; i++)
        {
            var panelSpec = spec.Panels[i];
            var panel = _panels[i];
            var plot = PlotAt(i).Plot;
            var row = i / spec.Columns;
            var col = i % spec.Columns;

            plot.Clear();
            panel.Bars.Clear();
            PlotStyle.Apply(plot, Theme);
            plot.Title(panelSpec.Title, size: Theme.TitleSize);
            plot.Axes.Bottom.Label.Text = row == rows - 1 ? spec.XLabel : string.Empty;
            plot.Axes.Left.Label.Text = col == 0 ? spec.YLabel : string.Empty;
            plot.Legend.IsVisible = i == 0 && panelSpec.Series.Count > 1;
            plot.Legend.Alignment = Alignment.UpperLeft;

            foreach (var band in spec.Bands)
            {
                var span = plot.Add.HorizontalSpan(band.Start, band.End);
                span.FillColor = ColorOf(band.Color).WithAlpha(Theme.HighContrast ? band.Alpha / 2 : band.Alpha);
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
                    var color = series.BarColors?[b] ?? series.Color ?? ChartColors.SessionA;
                    bars.Add(new Bar
                    {
                        Position = series.Centers[b] + offset,
                        Value = series.Values[b],
                        Size = barWidth,
                        FillColor = ColorOf(color).WithAlpha(PlotStyle.Alpha(Theme, n == 1 ? 1 : 0.85)),
                        LineWidth = 0,
                    });
                }

                var plottable = plot.Add.Bars(bars);
                plottable.LegendText = series.Label;
                panel.Bars.Add((series, offset, barWidth));
            }

            var zero = plot.Add.VerticalLine(0);
            zero.LineColor = ColorOf(Theme.Crosshair).WithAlpha(0.5);
            zero.LineWidth = 1;
            zero.LinePattern = LinePattern.Dashed;

            if (panelSpec.Annotation.Length > 0)
            {
                PlotStyle.Note(plot, Theme, panelSpec.Annotation, Alignment.UpperRight);
            }

            panel.Hover = PlotStyle.HoverLabel(plot, Theme);
            panel.Title = panelSpec.Title;

            plot.Axes.AutoScale();
            plot.Axes.SetLimitsX(spec.XLimits.Min, spec.XLimits.Max);
            plot.Axes.Bottom.Range.Min = spec.XLimits.Min;
            plot.Axes.Left.Range.Min = 0;
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

        // The bar under the pointer: the series whose offset bin contains x, and whose height exceeds y.
        foreach (var (series, offset, width) in panel.Bars)
        {
            for (var b = 0; b < series.Centers.Length; b++)
            {
                var center = series.Centers[b] + offset;
                if (at.X >= center - width / 2 && at.X <= center + width / 2 && at.Y >= 0 && at.Y <= series.Values[b])
                {
                    var who = series.Label.Length > 0 ? $"{panel.Title} - {series.Label}" : panel.Title;
                    var text = string.Format(CultureInfo.InvariantCulture, Spec.HoverFormat, who, series.Centers[b], series.Values[b]);
                    PlotStyle.ShowHover(panel.Hover, text, new Coordinates(center, series.Values[b]));
                    PlotAt(index).Refresh();
                    return;
                }
            }
        }

        HideHover(index);
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
        public List<(BarSeriesSpec Series, double Offset, double Width)> Bars { get; } = new();
    }
}
