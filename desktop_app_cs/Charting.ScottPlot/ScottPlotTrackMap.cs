using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>ScottPlot 5 implementation of a <see cref="TrackMapSpec"/>: coloured paths and labelled points in equal units.</summary>
public sealed class ScottPlotTrackMap : IChartSurface
{
    private readonly AvaPlot _avaPlot = new();
    private TrackMapSpec? _spec;
    private ChartTheme _theme = ChartTheme.Dark;

    /// <summary>Create the surface.</summary>
    public ScottPlotTrackMap()
    {
        DataAreaInput.Install(_avaPlot.UserInputProcessor);
        _avaPlot.PointerMoved += (_, e) => CursorMoved?.Invoke(ToCoordinates(e).X);
        _avaPlot.PointerExited += (_, _) => CursorMoved?.Invoke(null);
        _avaPlot.PointerPressed += (_, e) =>
        {
            var c = ToCoordinates(e);
            Clicked?.Invoke(new ChartPoint(c.X, c.Y, 0));
        };
    }

    /// <inheritdoc />
    public Control Control => _avaPlot;

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
        if (spec is not TrackMapSpec map)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(_spec, map))
        {
            return;
        }

        _spec = map;
        var plot = _avaPlot.Plot;
        plot.Clear();
        ScottPlotStripChart.ApplyTheme(plot, _theme);
        plot.Title(map.Title, size: 12);
        plot.Axes.Bottom.Label.Text = "X (m)";
        plot.Axes.Left.Label.Text = "Y (m)";
        plot.Legend.IsVisible = map.Paths.Any(p => p.Label.Length > 0);
        plot.Legend.Alignment = Alignment.UpperLeft;

        foreach (var path in map.Paths)
        {
            var line = plot.Add.ScatterLine(path.X, path.Y);
            line.Color = Color.FromHex(path.Color);
            line.LineWidth = (float)path.Width;
            line.LegendText = path.Label;
        }

        foreach (var point in map.Points)
        {
            var marker = plot.Add.Marker(point.X, point.Y, point.Star ? MarkerShape.FilledDiamond : MarkerShape.FilledCircle, (float)point.Size, Color.FromHex(point.Color));
            marker.MarkerLineColor = Color.FromHex(_theme.Text).WithAlpha(0.6);
            marker.MarkerLineWidth = 0.5f;
            if (point.Label.Length > 0)
            {
                var text = plot.Add.Text(point.Label, point.X, point.Y);
                text.LabelFontColor = Color.FromHex(_theme.Text);
                text.LabelFontSize = 10;
                text.OffsetX = 8;
                text.OffsetY = -8;
            }
        }

        plot.Axes.Rules.Clear();
        plot.Axes.AutoScale();
        plot.Axes.Rules.Add(new global::ScottPlot.AxisRules.SquareZoomOut(plot.Axes.Bottom, plot.Axes.Left));
        _avaPlot.Refresh();
    }

    /// <inheritdoc />
    public void SetCursor(double? x)
    {
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }

    private Coordinates ToCoordinates(PointerEventArgs e)
    {
        var position = e.GetPosition(_avaPlot);
        var scale = _avaPlot.DisplayScale;
        return _avaPlot.Plot.GetCoordinates(new Pixel((float)(position.X * scale), (float)(position.Y * scale)));
    }
}
