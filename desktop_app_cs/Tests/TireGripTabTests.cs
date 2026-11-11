using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The Tire Grip tab over the Fuji 86 sample.</summary>
public class TireGripTabTests
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

        window.ActiveTab = window.TireGrip;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TireGrip.RunNowAsync();
        return window;
    }

    [Fact]
    public async Task Tire_grip_tab_builds_four_panels_in_both_metric_modes()
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

        tab.Percentile = "90";
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
    }

    [Fact]
    public async Task Session_b_adds_a_second_series_and_comparison_tables()
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TireGrip.RunNowAsync();

        Assert.Equal(2, window.TireGrip.Spec!.Panels[0].Series.Count);
        Assert.Equal(2, window.TireGrip.Stats.Count);
    }

    [AvaloniaFact]
    public async Task Tire_grip_view_renders()
    {
        var viewModel = await LoadedWindow();
        var view = new TireGripView { DataContext = viewModel.TireGrip };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();

        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, "tire_grip.png"), new PngBitmapEncoderOptions());
    }
}
