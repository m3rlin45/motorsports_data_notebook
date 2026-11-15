using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Input;
using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Charting.ScottPlot;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// What the strip chart keeps and does between renders: the user's zoom when
/// the same thing is redrawn, the pinned cursor, and the keyboard.
/// </summary>
public class StripChartInteractionTests
{
    private static string FixturePath =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "laps.json");

    private static StripChartSpec Spec(string? viewKey, XAxisMode axis = XAxisMode.Distance) =>
        StripChartBuilder.Build(Fixture.Load(FixturePath), axis, viewKey);

    private static (Window Window, ScottPlotStripChart Chart) Show(StripChartSpec spec)
    {
        var chart = new ScottPlotStripChart();
        var window = new Window { Width = 1280, Height = 860, Content = chart.Control };
        window.Show();
        chart.Render(spec);
        window.CaptureRenderedFrame();
        return (window, chart);
    }

    private static (double Min, double Max) XRangeOf(ScottPlotStripChart chart, int panel = 0) =>
        (chart.Plots[panel].Axes.Bottom.Range.Min, chart.Plots[panel].Axes.Bottom.Range.Max);

    [Fact]
    public Task A_render_on_another_thread_does_not_touch_the_ui_from_there() => Ui.Run(async () =>
    {
        var (_, chart) = Show(Spec("session-1"));
        var plot = chart.Plots[0];
        plot.Axes.SetLimitsX(100, 900); // one panel zoomed; the others follow when its next render reports the change

        // On Windows the compositor draws the plots on its own thread, and ScottPlot reports changed
        // limits from inside that render. Reacting there (invalidating a control) throws in Avalonia's
        // render loop: the frame is lost, and the window stopped drawing after a tab switch.
        await Task.Run(() =>
        {
            using var surface = SkiaSharp.SKSurface.Create(new SkiaSharp.SKImageInfo(640, 240));
            plot.Render(surface.Canvas, 640, 240);
        }, TestContext.Current.CancellationToken);

        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal((100, 900), XRangeOf(chart, chart.Plots.Count - 1)); // synchronised, on the UI thread
    });

    [Fact]
    public Task Composing_a_frame_replays_the_chart_without_running_scottplot() => Ui.Run(() =>
    {
        var (window, chart) = Show(Spec("session-1"));
        var plot = chart.Plots[0];
        var renders = plot.RenderManager.RenderCount;
        Assert.True(renders > 0);

        // The cursor overlay lies over the plot: redrawing it makes the compositor draw those pixels
        // again, the plot's among them. With ScottPlot's own control that ran ScottPlot's whole render
        // on the compositor's thread; UiThreadPlot hands the compositor a recorded picture instead.
        chart.Overlays[0].InvalidateVisual();
        window.CaptureRenderedFrame();
        Assert.Equal(renders, plot.RenderManager.RenderCount);

        // The plot itself invalidated: one render, on this (the UI) thread.
        chart.PinCursor(2000);
        chart.ZoomAboutCursor(0.5);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        window.CaptureRenderedFrame();
        Assert.True(plot.RenderManager.RenderCount > renders);
    });

    [Fact]
    public Task Redrawing_the_same_view_keeps_the_zoom_and_a_new_view_resets_it() => Ui.Run(() =>
    {
        var (_, chart) = Show(Spec("session-1"));
        var full = XRangeOf(chart);

        chart.PinCursor(2000);
        chart.ZoomAboutCursor(0.25);
        var zoomed = XRangeOf(chart);
        Assert.True(zoomed.Max - zoomed.Min < (full.Max - full.Min) / 3);

        // A lap or channel ticked: a new spec for the same session.
        chart.Render(Spec("session-1"));
        Assert.Equal(zoomed, XRangeOf(chart));
        Assert.Equal(zoomed, XRangeOf(chart, chart.Plots.Count - 1)); // every panel
        Assert.Equal(2000, chart.Cursor);
        Assert.True(chart.IsCursorPinned);

        // A theme change redraws the same spec.
        chart.Theme = ChartTheme.Light;
        Assert.Equal(zoomed, XRangeOf(chart));

        // Another session: back to the whole lap, cursor gone.
        chart.Render(Spec("session-2"));
        Assert.Equal(full, XRangeOf(chart));
        Assert.Null(chart.Cursor);

        // The other x-axis is a different view even for the same session.
        chart.ZoomAboutCursor(0.25);
        chart.Render(Spec("session-2", XAxisMode.Time));
        Assert.True(XRangeOf(chart).Max < 200); // seconds, not metres

        // Specs without a key never carry the view over.
        chart.Render(Spec(null));
        chart.ZoomAboutCursor(0.25);
        chart.Render(Spec(null));
        Assert.Equal(full, XRangeOf(chart));
    });

    [Fact]
    public Task Bringing_a_region_into_view_shades_it_and_keeps_a_pinned_cursor_that_is_inside_it() => Ui.Run(() =>
    {
        var fixture = Fixture.Load(FixturePath);
        var whole = StripChartBuilder.Build(fixture, XAxisMode.Distance, "session-1");
        var lap = whole.InitialView!.Value;
        var width = lap.Max - lap.Min;
        StripChartSpec Focused(string name, double from, double to) => StripChartBuilder.Build(
            fixture, XAxisMode.Distance, "session-1", new[] { new StripChartBuilder.RegionSpan(name, new XRange(0, 1), new XRange(lap.Min + from * width, lap.Min + to * width)) });

        var (window, chart) = Show(whole);
        Assert.All(chart.Plots, p => Assert.Empty(p.GetPlottables<ScottPlot.Plottables.HorizontalSpan>()));
        chart.PinCursor(lap.Min + 0.35 * width);

        // A region around the cursor: every panel shades it, the view goes to it, the cursor stays put.
        chart.Render(Focused("Turn 3", 0.30, 0.40));
        window.CaptureRenderedFrame();
        Assert.All(chart.Plots, p => Assert.Single(p.GetPlottables<ScottPlot.Plottables.HorizontalSpan>()));
        var view = XRangeOf(chart);
        Assert.InRange(view.Min, lap.Min + 0.25 * width, lap.Min + 0.30 * width);
        Assert.InRange(view.Max, lap.Min + 0.40 * width, lap.Min + 0.45 * width);
        Assert.True(chart.IsCursorPinned);
        Assert.Equal(lap.Min + 0.35 * width, chart.Cursor!.Value, 6);

        // A region elsewhere: a cursor left behind out of sight would be a readout of nothing on screen.
        chart.Render(Focused("Turn 7", 0.70, 0.80));
        window.CaptureRenderedFrame();
        Assert.False(chart.IsCursorPinned);
        Assert.Null(chart.Cursor);
    });

    [Fact]
    public Task Reset_view_returns_to_the_initial_view() => Ui.Run(() =>
    {
        var (_, chart) = Show(Spec("s"));
        var full = XRangeOf(chart);
        chart.ZoomAboutCursor(0.1);
        Assert.NotEqual(full, XRangeOf(chart));

        chart.ResetView();

        Assert.Equal(full, XRangeOf(chart));
    });

    [Fact]
    public Task A_pinned_cursor_ignores_the_pointer_and_steps_by_sample() => Ui.Run(() =>
    {
        var (window, chart) = Show(Spec("s"));
        var moves = new List<double?>();
        chart.CursorMoved += moves.Add;

        chart.PinCursor(1000);
        window.MouseMove(new Avalonia.Point(900, 300));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(1000, chart.Cursor); // the pointer is free to go and tick a lap

        chart.StepCursor(1);
        var one = chart.Cursor!.Value;
        Assert.True(one > 1000);
        chart.StepCursor(10);
        Assert.True(chart.Cursor > one);
        chart.StepCursor(-11);
        Assert.InRange(chart.Cursor!.Value, 990, 1010);
        Assert.Equal(chart.Cursor, moves[^1]); // the linked surfaces follow

        chart.UnpinCursor();
        Assert.False(chart.IsCursorPinned);
        Assert.Null(chart.Cursor);
        Assert.Null(moves[^1]);

        // Unpinned, the cursor follows the pointer again.
        window.MouseMove(new Avalonia.Point(900, 300));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.NotNull(chart.Cursor);
    });

    [Fact]
    public Task Stepping_past_the_view_scrolls_it_to_keep_the_cursor_visible() => Ui.Run(() =>
    {
        var (_, chart) = Show(Spec("s"));
        chart.PinCursor(2000);
        chart.ZoomAboutCursor(0.01); // a few tens of metres
        var before = XRangeOf(chart);

        chart.StepCursor(200);

        var after = XRangeOf(chart);
        Assert.True(after.Min > before.Min);
        Assert.InRange(chart.Cursor!.Value, after.Min, after.Max);
        Assert.Equal(before.Max - before.Min, after.Max - after.Min, 6); // scrolled, not zoomed
    });

    [Fact]
    public Task A_click_pins_the_cursor_and_the_keys_drive_it() => Ui.Run(() =>
    {
        var (window, chart) = Show(Spec("s"));

        window.MouseDown(new Avalonia.Point(640, 120), MouseButton.Left);
        window.MouseUp(new Avalonia.Point(640, 120), MouseButton.Left);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.True(chart.IsCursorPinned);
        var pinned = chart.Cursor!.Value;

        window.KeyPressQwerty(PhysicalKey.ArrowRight, RawInputModifiers.None);
        Assert.True(chart.Cursor > pinned);

        var span = XRangeOf(chart).Max - XRangeOf(chart).Min;
        window.KeyPressQwerty(PhysicalKey.ArrowUp, RawInputModifiers.None);
        Assert.True(XRangeOf(chart).Max - XRangeOf(chart).Min < span);

        window.KeyPressQwerty(PhysicalKey.Home, RawInputModifiers.None);
        Assert.Equal(span, XRangeOf(chart).Max - XRangeOf(chart).Min, 6);

        window.KeyPressQwerty(PhysicalKey.Escape, RawInputModifiers.None);
        Assert.False(chart.IsCursorPinned);
    });

    [Fact]
    public Task Readouts_name_the_lap_and_carry_units() => Ui.Run(() =>
    {
        var (_, chart) = Show(Spec("s"));

        chart.SetCursor(2158);

        var speed = chart.Overlays[0].Readouts;
        Assert.Equal(2, speed.Count);
        Assert.StartsWith("L14  ", speed[0].Text);
        Assert.EndsWith(" km/h", speed[0].Text);
        Assert.StartsWith("L12  ", speed[1].Text);
        // The cursor's x position is labelled on the bottom panel only.
        Assert.Equal(string.Empty, chart.Overlays[0].XLabel);
        Assert.Equal("2158 m", chart.Overlays[^1].XLabel);

        chart.SetCursor(null);
        Assert.Empty(chart.Overlays[0].Readouts);
    });
}
