using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Renders the Time/Distance view headlessly against the committed fixture
/// and writes the frames to <c>bin/.../renders/</c>. Proves the ScottPlot
/// adapter draws without a display and leaves images to eyeball.
/// </summary>
public class RenderSmokeTests
{
    private static string FixturePath =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", "laps.json");

    /// <summary>Where the PNGs go.</summary>
    public static string RenderDir
    {
        get
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "renders");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    /// <summary>A window showing the Time/Distance tab for a fixture.</summary>
    private static (Window Window, TimeDistanceViewModel ViewModel, TimeDistanceView View) ShowFixture(string path, double width = 1280, double height = 860)
    {
        var vm = new TimeDistanceViewModel(new SessionViewModel("A", new ProfileStore("/nonexistent/profiles.yaml")));
        var view = new TimeDistanceView { DataContext = vm };
        var window = new Window { Width = width, Height = height, Content = view, Background = Avalonia.Media.Brushes.Black };
        window.Show();
        vm.ShowFixture(Fixture.Load(path));
        return (window, vm, view);
    }

    [AvaloniaFact]
    public void Strip_chart_renders_distance_and_time_modes()
    {
        Assert.True(File.Exists(FixturePath), $"fixture missing: {FixturePath}");
        var (window, vm, _) = ShowFixture(FixturePath);

        // Park the cursor mid-lap so the frame shows the crosshair and value labels too.
        vm.Cursors.SetCursor(2158.0);
        var distance = window.CaptureRenderedFrame();
        Assert.NotNull(distance);
        distance!.Save(Path.Combine(RenderDir, "strip_chart_distance.png"), new PngBitmapEncoderOptions());

        vm.IsTime = true;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        vm.Cursors.SetCursor(60.0);
        var time = window.CaptureRenderedFrame();
        Assert.NotNull(time);
        time!.Save(Path.Combine(RenderDir, "strip_chart_time.png"), new PngBitmapEncoderOptions());

        Assert.StartsWith("2 laps", vm.Summary);
        Assert.Equal(XAxisMode.Time, vm.Spec!.XAxis);
    }

    /// <summary>
    /// Opt-in stress render: set INFERNO_DENSE_FIXTURE to a fixture path
    /// (e.g. the output of `just cs-fixture-dense`) to render it and report
    /// load/render timings in the test output. Skipped otherwise.
    /// </summary>
    [AvaloniaFact]
    public void Dense_fixture_renders_when_provided()
    {
        var path = Environment.GetEnvironmentVariable("INFERNO_DENSE_FIXTURE");
        Assert.SkipWhen(string.IsNullOrEmpty(path) || !File.Exists(path), "set INFERNO_DENSE_FIXTURE to a dense fixture path");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (window, vm, _) = ShowFixture(path!);
        var loadMs = sw.ElapsedMilliseconds;
        var frame = window.CaptureRenderedFrame();
        var renderMs = sw.ElapsedMilliseconds - loadMs;

        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderDir, "strip_chart_dense.png"), new PngBitmapEncoderOptions());
        File.WriteAllText(Path.Combine(RenderDir, "strip_chart_dense.txt"), $"{vm.Summary}\nload+build {loadMs} ms, first render {renderMs} ms\n");
    }

    /// <summary>
    /// Renders the ScottPlot backend directly with the view panned to the
    /// start line, so the frame shows the previous lap faded on the left, the
    /// lap boundary markers, and the cursor inside the context region.
    /// </summary>
    [AvaloniaFact]
    public void Context_laps_render_faded_with_lap_markers()
    {
        var fixture = Fixture.Load(FixturePath);
        var spec = StripChartBuilder.Build(fixture, XAxisMode.Distance) with
        {
            InitialView = new XRange(-1500, 1500),
        };

        using var surface = ChartSurfaces.CreateStripChart();
        var window = new Window { Width = 1280, Height = 860, Content = surface.Control, Background = Avalonia.Media.Brushes.Black };
        window.Show();
        surface.Render(spec);
        surface.SetCursor(-400);

        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderDir, "strip_chart_context.png"), new PngBitmapEncoderOptions());
    }

    /// <summary>
    /// Drives a real pointer move through Avalonia's input pipeline (not
    /// <see cref="CursorLinkGroup.SetCursor"/>) so the hover path from the
    /// ScottPlot control to the link group is exercised end to end.
    /// </summary>
    [AvaloniaFact]
    public void Hovering_a_panel_moves_the_linked_cursor()
    {
        var (window, vm, _) = ShowFixture(FixturePath);
        window.CaptureRenderedFrame(); // ScottPlot needs one render before it can map pixels to data

        var moves = new List<double?>();
        vm.Cursors.CursorChanged += moves.Add;

        // The channel column on the left is 200 px wide; x = 640 is well inside the chart.
        window.MouseMove(new Point(640, 300));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var x = Assert.Single(moves);
        Assert.NotNull(x);
        Assert.InRange(x!.Value, 1500, 3000); // roughly mid-lap on the 4.5 km Fuji fixture
        Assert.Equal(x, vm.Cursors.Cursor);
    }
}
