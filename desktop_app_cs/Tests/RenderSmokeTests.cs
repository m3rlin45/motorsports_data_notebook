using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Renders the spike window headlessly against the committed fixture and
/// writes the frame to <c>bin/.../renders/</c>. Proves the ScottPlot adapter
/// draws without a display and leaves an image to eyeball.
/// </summary>
public class RenderSmokeTests
{
    private static string FixturePath =>
        Path.Combine(AppContext.BaseDirectory, "fixtures", "laps.json");

    private static string RenderDir
    {
        get
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "renders");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    [AvaloniaFact]
    public void Strip_chart_renders_distance_and_time_modes()
    {
        Assert.True(File.Exists(FixturePath), $"fixture missing: {FixturePath}");

        var window = new MainWindow(FixturePath) { Width = 1280, Height = 860 };
        window.Show();

        var distance = window.CaptureRenderedFrame();
        Assert.NotNull(distance);
        distance!.Save(Path.Combine(RenderDir, "strip_chart_distance.png"), new PngBitmapEncoderOptions());

        window.TimeRadio.IsChecked = true;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        var time = window.CaptureRenderedFrame();
        Assert.NotNull(time);
        time!.Save(Path.Combine(RenderDir, "strip_chart_time.png"), new PngBitmapEncoderOptions());

        Assert.StartsWith("CMD_Inferno", window.FixtureLabel.Text);
        Assert.Contains("2 laps", window.FixtureLabel.Text);
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
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return;
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var window = new MainWindow(path) { Width = 1280, Height = 860 };
        var loadMs = sw.ElapsedMilliseconds;
        window.Show();
        var frame = window.CaptureRenderedFrame();
        var renderMs = sw.ElapsedMilliseconds - loadMs;

        Assert.NotNull(frame);
        var output = Path.Combine(RenderDir, "strip_chart_dense.png");
        frame!.Save(output, new PngBitmapEncoderOptions());
        File.WriteAllText(
            Path.Combine(RenderDir, "strip_chart_dense.txt"),
            $"{window.FixtureLabel.Text}\nload+build {loadMs} ms, first render {renderMs} ms\n");
    }

    [AvaloniaFact]
    public void Cursor_moves_update_the_readout()
    {
        var window = new MainWindow(FixturePath);
        window.Show();

        window.Cursors.SetCursor(2158.0);

        Assert.StartsWith("2158 m | Speed: ", window.Readout.Text);
    }
}
