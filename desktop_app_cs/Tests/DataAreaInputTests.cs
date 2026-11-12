using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Charting.ScottPlot;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Drives mouse gestures through Avalonia's headless input into the ScottPlot
/// control and checks which axes moved. The window holds only the chart, so
/// window points are chart points; the top panel's data area begins 72 px
/// from the left (the y-axis strip) and 36 px from the top.
/// </summary>
public class DataAreaInputTests
{
    private static string FixturePath =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", "laps.json");

    private static (Window Window, ScottPlotStripChart Chart) Show()
    {
        var chart = new ScottPlotStripChart();
        var window = new Window { Width = 1280, Height = 860, Content = chart.Control };
        window.Show();
        chart.Render(StripChartBuilder.Build(Fixture.Load(FixturePath), XAxisMode.Distance));
        window.CaptureRenderedFrame(); // ScottPlot needs a render before it can hit-test axes
        return (window, chart);
    }

    /// <summary>Snapshot of an axis range; ScottPlot's <c>Range</c> is a live mutable object.</summary>
    private static (double Min, double Max, double Span) Snapshot(global::ScottPlot.IAxis axis) =>
        (axis.Range.Min, axis.Range.Max, axis.Range.Span);

    private static void Drag(Window window, Point from, Point to, MouseButton button = MouseButton.Left)
    {
        window.MouseDown(from, button);
        window.MouseMove(new Point((from.X + to.X) / 2, (from.Y + to.Y) / 2));
        window.MouseMove(to);
        window.MouseUp(to, button);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
    }

    [AvaloniaFact]
    public void Dragging_the_data_area_pans_x_only()
    {
        var (window, chart) = Show();
        var plot = chart.Plots[0];
        var x0 = Snapshot(plot.Axes.Bottom);
        var y0 = Snapshot(plot.Axes.Left);

        Drag(window, new Point(640, 120), new Point(540, 60));

        Assert.NotEqual(x0.Min, plot.Axes.Bottom.Range.Min);
        Assert.Equal(y0.Min, plot.Axes.Left.Range.Min);
        Assert.Equal(y0.Max, plot.Axes.Left.Range.Max);
        // The x pan is mirrored to the other panels.
        Assert.Equal(plot.Axes.Bottom.Range.Min, chart.Plots[^1].Axes.Bottom.Range.Min);
    }

    [AvaloniaFact]
    public void Dragging_the_y_axis_pans_y()
    {
        var (window, chart) = Show();
        var plot = chart.Plots[0];
        var y0 = Snapshot(plot.Axes.Left);

        Drag(window, new Point(30, 120), new Point(30, 60));

        Assert.NotEqual(y0.Min, plot.Axes.Left.Range.Min);
    }

    [AvaloniaFact]
    public void Wheel_over_the_data_area_zooms_x_only()
    {
        var (window, chart) = Show();
        var plot = chart.Plots[0];
        var x0 = Snapshot(plot.Axes.Bottom);
        var y0 = Snapshot(plot.Axes.Left);

        window.MouseWheel(new Point(640, 120), new Vector(0, 1));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        Assert.True(plot.Axes.Bottom.Range.Span < x0.Span);
        Assert.Equal(y0.Min, plot.Axes.Left.Range.Min);
        Assert.Equal(y0.Max, plot.Axes.Left.Range.Max);
    }

    [AvaloniaFact]
    public void Wheel_over_the_y_axis_zooms_y()
    {
        var (window, chart) = Show();
        var plot = chart.Plots[0];
        var y0 = Snapshot(plot.Axes.Left);

        window.MouseWheel(new Point(30, 120), new Vector(0, 1));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        Assert.True(plot.Axes.Left.Range.Span < y0.Span);
    }
}
