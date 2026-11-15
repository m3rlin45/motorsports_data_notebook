using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;
using ScottPlot.Plottables;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot 5 implementation of a <see cref="BoxPlotSpec"/>: stacked panels
/// of boxes with whiskers, a dashed mean line and flier markers, shaded
/// category bands, and a hover tooltip with the box statistics. A click
/// reports the category under the pointer through <see cref="ChartPoint.ItemId"/>.
/// </summary>
public sealed class ScottPlotBoxPlot : ScottPlotGridSurface<BoxPlotSpec>
{
    private static readonly Avalonia.Input.Cursor HandCursor = new(StandardCursorType.Hand);
    private const float PadLeft = 72;
    private const float PadRight = 16;
    private const float PadTop = 30;
    private const float PadBottomLast = 90;
    private const float PadBottomInner = 14;

    private readonly List<Panel> _panels = new();

    /// <inheritdoc />
    protected override void OnPlotCreated(AvaPlot avaPlot, int index) => _panels.Add(new Panel());

    /// <inheritdoc />
    protected override void Build(BoxPlotSpec spec, BoxPlotSpec? previous)
    {
        EnsurePlots(spec.Panels.Count, 1);
        _panels.RemoveRange(PlotCount, _panels.Count - PlotCount);
        var ink = ColorOf(Theme.Text);

        for (var i = 0; i < spec.Panels.Count; i++)
        {
            var panelSpec = spec.Panels[i];
            var panel = _panels[i];
            var plot = PlotAt(i).Plot;
            var isBottom = i == spec.Panels.Count - 1;

            plot.Clear();
            panel.Boxes.Clear();
            PlotStyle.Apply(plot, Theme);
            // Named arguments on purpose: PixelPadding's positional order is (left, right, bottom, top).
            plot.Layout.Fixed(new PixelPadding(left: PadLeft, right: PadRight, bottom: isBottom ? PadBottomLast : PadBottomInner, top: PadTop));
            plot.Title(panelSpec.Title, size: Theme.TitleSize);
            plot.Axes.Left.Label.Text = panelSpec.YLabel;
            plot.Axes.Bottom.TickLabelStyle.IsVisible = isBottom;
            plot.Legend.IsVisible = i == 0 && spec.Legend.Count > 0;
            plot.Legend.Alignment = Alignment.UpperRight;

            foreach (var band in spec.Bands)
            {
                var span = plot.Add.HorizontalSpan(band.Start, band.End);
                span.FillColor = ColorOf(band.Color).WithAlpha(band.Alpha);
                span.LineWidth = 0;
            }

            if (panelSpec.ZeroLine)
            {
                var zero = plot.Add.HorizontalLine(0);
                zero.LineColor = ink.WithAlpha(0.3);
                zero.LineWidth = 1;
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
                plottable.FillColor = ColorOf(b.Color).WithAlpha(PlotStyle.Alpha(Theme, 0.7));
                plottable.LineColor = ink;
                plottable.LineWidth = 1;
                panel.Boxes.Add(b);
            }

            foreach (var b in panelSpec.Boxes)
            {
                // Median as a solid ink line, mean dashed: both read on either theme and on any box fill.
                var median = plot.Add.Line(b.Position - b.Width / 2, b.Median, b.Position + b.Width / 2, b.Median);
                median.Color = ink;
                median.LineWidth = 2;
                var mean = plot.Add.Line(b.Position - b.Width / 2, b.Mean, b.Position + b.Width / 2, b.Mean);
                mean.Color = ink;
                mean.LineWidth = 1;
                mean.LinePattern = LinePattern.Dashed;
                if (b.Fliers.Length > 0)
                {
                    var fliers = plot.Add.Markers(Enumerable.Repeat(b.Position, b.Fliers.Length).ToArray(), b.Fliers, MarkerShape.OpenCircle, 5, ink.WithAlpha(0.7));
                    fliers.MarkerLineWidth = 1;
                }
            }

            if (i == 0)
            {
                foreach (var (label, color) in spec.Legend)
                {
                    plot.Legend.ManualItems.Add(new LegendItem { LabelText = label, FillColor = ColorOf(color).WithAlpha(PlotStyle.Alpha(Theme, 0.7)) });
                }
            }

            plot.Axes.Bottom.SetTicks(spec.CategoryPositions, spec.CategoryLabels);
            plot.Axes.Bottom.TickLabelStyle.Rotation = -45;
            plot.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleRight;

            panel.Hover = PlotStyle.HoverLabel(plot, Theme);

            plot.Axes.AutoScale();
            var n = spec.CategoryPositions.Length;
            if (n > 0)
            {
                plot.Axes.SetLimitsX(spec.CategoryPositions[0] - 0.75, spec.CategoryPositions[n - 1] + 0.75);
            }
        }

        CommitView();
    }

    /// <summary>Index of the category whose column contains <paramref name="x"/>, or -1 between or outside columns.</summary>
    internal int CategoryAt(double x)
    {
        if (Spec is null)
        {
            return -1;
        }

        for (var i = 0; i < Spec.CategoryPositions.Length; i++)
        {
            if (Math.Abs(x - Spec.CategoryPositions[i]) <= 0.5)
            {
                return i;
            }
        }

        return -1;
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(int index, Coordinates at, PointerPressedEventArgs e) =>
        RaiseClicked(new ChartPoint(at.X, at.Y, index, CategoryAt(at.X)));

    /// <inheritdoc />
    protected override void OnPointerMoved(int index, Coordinates at, PointerEventArgs e)
    {
        base.OnPointerMoved(index, at, e);
        PlotAt(index).Cursor = CategoryAt(at.X) >= 0 ? HandCursor : Avalonia.Input.Cursor.Default;
        var panel = _panels[index];
        if (panel.Hover is null)
        {
            return;
        }

        foreach (var b in panel.Boxes)
        {
            if (Math.Abs(at.X - b.Position) <= b.Width / 2 && at.Y >= b.WhiskerLow && at.Y <= b.WhiskerHigh && b.Tooltip.Length > 0)
            {
                PlotStyle.ShowHover(panel.Hover, b.Tooltip, at);
                PlotAt(index).Refresh();
                return;
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
        public Text? Hover { get; set; }
        public List<BoxSpec> Boxes { get; } = new();
    }
}
