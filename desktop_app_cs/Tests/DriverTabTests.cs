using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The Driver Consistency tab over the Fuji 86 sample.</summary>
public class DriverTabTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static async Task<MainWindowViewModel> LoadedWindow()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        foreach (var tab in window.Tabs)
        {
            tab.Debounce = TimeSpan.Zero;
        }

        window.ActiveTab = window.Driver;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.Driver.RunNowAsync();
        return window;
    }

    [Fact]
    public async Task Driver_tab_builds_summary_detail_and_map_with_the_auto_throttle_threshold()
    {
        var window = await LoadedWindow();
        var tab = window.Driver;

        Assert.StartsWith("Done - ", tab.Status);
        Assert.Contains("TA pts", tab.Status);
        // 95 % of the throttle channel's 95th percentile: PPS tops out near 100.
        var threshold = double.Parse(tab.DriverConfig.ThresholdsA.ThrottleThreshold, System.Globalization.CultureInfo.InvariantCulture);
        Assert.InRange(threshold, 85, 100);

        var summary = tab.SummarySpec!;
        Assert.Equal(3, summary.Panels.Count);
        Assert.Equal(tab.Corners.Count, summary.CategoryLabels.Length);
        Assert.True(summary.Panels[1].Boxes.Count > 0); // throttle acceptance boxes
        Assert.True(summary.Bands.Count is >= 1 and <= 3); // top-opportunity bands

        Assert.NotNull(tab.MapSpec);
        Assert.True(tab.MapSpec!.Paths.Count > 1);
        Assert.Contains(tab.MapSpec.Points, p => p.Star);

        Assert.Equal(DriverViewMode.Summary, tab.ViewMode);
        tab.SelectedCorner = tab.Corners[1];
        Assert.Equal(DriverViewMode.Detail, tab.ViewMode);
        var detail = tab.DetailSpec!;
        Assert.True(detail.Legend); // the detail overlay has no lap list, so the in-plot legend names the laps
        Assert.Equal(3, detail.Subplots.Count);
        Assert.Equal(window.SessionA.SelectedLaps.Count, detail.Subplots[0].Traces.Count);
        Assert.Contains(detail.Markers!, m => m.Label == "Apex");

        Assert.Equal(2, tab.Stats.Count);
        Assert.Equal(tab.Corners.Count, tab.Stats[0].Rows.Count);
    }

    [Fact]
    public void Box_statistics_follow_numpy_and_matplotlib_conventions()
    {
        var box = BoxSpec.From(1, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 }, "#000000", "tip");

        Assert.Equal(3.25, box.Q1);
        Assert.Equal(5.5, box.Median);
        Assert.Equal(7.75, box.Q3);
        Assert.Equal(1, box.WhiskerLow);
        Assert.Equal(9, box.WhiskerHigh);
        Assert.Equal(new double[] { 100 }, box.Fliers);
        Assert.Equal(14.5, box.Mean);
        Assert.Contains("Median:  5.50", DriverChartBuilder.BoxStatsText(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 }, "Turn 1 (L)", "%"));
    }

    [AvaloniaFact]
    public async Task Driver_views_render()
    {
        var viewModel = await LoadedWindow();
        var view = new DriverView { DataContext = viewModel.Driver };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();

        foreach (var (mode, name) in new[] { (DriverViewMode.Summary, "summary"), (DriverViewMode.Detail, "detail"), (DriverViewMode.Map, "map") })
        {
            viewModel.Driver.ViewMode = mode;
            if (mode == DriverViewMode.Detail)
            {
                viewModel.Driver.SelectedCorner = viewModel.Driver.Corners[0];
            }

            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            var frame = window.CaptureRenderedFrame();
            Assert.NotNull(frame);
            frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, $"driver_{name}.png"), new PngBitmapEncoderOptions());
        }
    }
}
