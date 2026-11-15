using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The Tire Grip tab over the Fuji 86 sample.</summary>
public class TireGripTabTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly InfernoAnalyzer.Core.Tracks.TrackStore NoTracks = new("/nonexistent/tracks.yaml"); // not this machine's saved tracks
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static async Task<MainWindowViewModel> LoadedWindow()
    {
        var window = new MainWindowViewModel(BuiltinOnly, NoTracks);
        foreach (var tab in window.Tabs)
        {
            tab.Debounce = TimeSpan.Zero;
        }

        window.ActiveTab = window.TireGrip;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TireGrip.RunNowAsync();
        return window;
    }

    [Fact]
    public Task Tire_grip_tab_builds_four_panels_in_both_metric_modes() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.TireGrip;

        Assert.Equal("Analysis complete", tab.Status);
        var spec = tab.Spec!;
        Assert.Equal(4, spec.Panels.Count);
        Assert.StartsWith("Pressure (", spec.XLabel);
        Assert.True(spec.Panels[0].Series[0].X.Length > 3);
        Assert.StartsWith("n = ", spec.Panels[0].Annotation);
        Assert.Single(tab.Stats);

        tab.Percentile.Text = "90";
        await tab.RunNowAsync();
        var p90 = tab.Spec!.Panels[0].Series[0];
        // Same buckets (the edges depend on the metric only); a lower percentile never exceeds a higher one.
        Assert.Equal(spec.Panels[0].Series[0].X, p90.X);
        for (var i = 0; i < p90.Y.Length; i++)
        {
            Assert.True(p90.Y[i] <= spec.Panels[0].Series[0].Y[i] + 1e-9);
        }

        tab.IsTemperature = true;
        await tab.RunNowAsync();
        Assert.StartsWith("Temperature (", tab.Spec!.XLabel);
    });

    [Fact]
    public Task Session_b_adds_a_second_series_and_comparison_tables() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TireGrip.RunNowAsync();

        Assert.Equal(2, window.TireGrip.Spec!.Panels[0].Series.Count);
        Assert.Equal(2, window.TireGrip.Stats.Count);
    });

    [Fact]
    public Task Tire_grip_view_renders() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var view = new TireGripView { DataContext = viewModel.TireGrip };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();

        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        RenderSmokeTests.SaveRender(frame!, "tire_grip.png");
    });

    [Fact]
    public Task A_ticked_region_restricts_the_grip_curves_with_the_whole_lap_curve_for_reference() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        await viewModel.Regions.LastRefresh; // started by the session load
        var tab = viewModel.TireGrip;
        viewModel.ActiveTab = tab;
        await tab.RunNowAsync();
        var wholeCounts = tab.Spec!.Panels[0].Series[0].Counts.Sum();
        var wholeTitle = tab.Spec.Title;

        viewModel.Regions.Corners[2].IsSelected = true;
        viewModel.Regions.Corners[4].IsSelected = true;
        await tab.RunNowAsync();

        var panel = tab.Spec!.Panels[0];
        Assert.EndsWith(" · Turn 3 + Turn 5", tab.Spec.Title);
        Assert.Equal(new[] { "Whole lap", "Selected regions" }, panel.Series.Select(s => s.Label));
        Assert.Equal(InfernoAnalyzer.Charting.ChartColors.Neutral, panel.Series[0].Color);
        Assert.Equal(wholeCounts, panel.Series[0].Counts.Sum());
        Assert.InRange(panel.Series[1].Counts.Sum(), 1, wholeCounts - 1); // the two corners' samples only

        var view = new TireGripView { DataContext = tab };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "tire_grip_region.png");

        viewModel.Regions.Clear();
        await tab.RunNowAsync();
        Assert.Equal(wholeTitle, tab.Spec!.Title);
        Assert.Single(tab.Spec.Panels[0].Series);
    });

    [Fact]
    public Task Session_b_without_laps_is_not_compared_and_the_percentile_is_validated() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.TireGrip;
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.SelectNone();
        await tab.RunNowAsync();
        Assert.Equal("Analysis complete", tab.Status);
        Assert.Single(tab.Spec!.Panels[0].Series);

        tab.Percentile.Text = "150";
        await tab.RunNowAsync();
        Assert.Contains("Percentile must be", tab.Status);
        Assert.Null(tab.Spec);

        tab.Percentile.Text = "99";
        await tab.RunNowAsync();
        Assert.Equal("Analysis complete", tab.Status);
        Assert.Contains("Pressure: {1:0.0}", tab.Spec!.HoverFormat); // the hover names what the x axis is
    });

    [Fact]
    public Task The_last_tab_and_its_settings_come_back_on_the_next_launch() => Ui.Run(() =>
    {
        var dir = Directory.CreateTempSubdirectory("inferno-settings");
        try
        {
            var path = Path.Combine(dir.FullName, "analyzer_settings.json");
            var first = new MainWindowViewModel(BuiltinOnly, settings: new SettingsStore(path));
            first.ActiveTab = first.TireGrip;
            first.TireGrip.Metric = TireMetric.Temperature;
            first.TireGrip.Percentile.Text = "95";
            first.TireGrip.ShowStats = true;
            first.SaveSettings();

            var second = new MainWindowViewModel(BuiltinOnly, settings: new SettingsStore(path));
            Assert.Same(second.TireGrip, second.ActiveTab);
            Assert.Equal(TireMetric.Temperature, second.TireGrip.Metric);
            Assert.Equal(95, second.TireGrip.Percentile.Value);
            Assert.True(second.TireGrip.ShowStats);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });
}
