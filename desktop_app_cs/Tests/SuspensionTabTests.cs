using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using Avalonia.Styling;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The Suspension tab over the Fuji 86 sample, plus the theme following.</summary>
public class SuspensionTabTests
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

        window.ActiveTab = window.Suspension;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.Suspension.RunNowAsync();
        return window;
    }

    [Fact]
    public async Task Suspension_tab_builds_four_panels_and_stats_from_the_profile_config()
    {
        var window = await LoadedWindow();
        var tab = window.Suspension;

        Assert.Equal("Analysis complete", tab.Status);
        Assert.Equal("LF_Shock_Pot", tab.SuspensionConfig.EntriesA.First(e => e.Key == "shock_fl").Value);
        Assert.Equal("0.997", tab.SuspensionConfig.RatiosA.FrontLeft);
        Assert.Equal("0.768", tab.SuspensionConfig.RatiosA.RearRight);

        var spec = tab.Spec!;
        Assert.Equal(4, spec.Panels.Count);
        Assert.Equal(2, spec.Columns);
        Assert.Equal("Front Left (FL)", spec.Panels[0].Title);
        Assert.Single(spec.Panels[0].Series);
        Assert.Equal(61, spec.Panels[0].Series[0].Centers.Length);
        Assert.Equal(100.0, spec.Panels[0].Series[0].Values.Sum(), 1e-9);
        Assert.StartsWith("Skew:", spec.Panels[0].Annotation);
        Assert.Equal(7, spec.Bands.Count);

        Assert.Equal(3, tab.Stats.Count);
        Assert.Equal(4, tab.Stats[0].Rows.Count);
        Assert.Equal("FL", tab.Stats[0].Rows[0][0]);
        Assert.Equal(2, tab.Stats[2].Rows.Count);
    }

    [Fact]
    public async Task Changing_a_channel_entry_or_ratio_reruns_the_analysis()
    {
        var window = await LoadedWindow();
        var tab = window.Suspension;
        var runs = 0;
        tab.AnalysisCompleted += () => runs++;

        tab.SuspensionConfig.RatiosA.FrontLeft = "0.5";
        await tab.RunNowAsync();
        Assert.True(runs >= 1);
        var afterRatio = tab.Spec!;

        tab.SuspensionConfig.EntriesA.First(e => e.Key == "shock_fl").Value = "does_not_exist";
        await tab.RunNowAsync();
        // The multi-lap analysis skips laps it cannot process; with every lap failing the result is empty.
        Assert.Equal(0.0, tab.Spec!.Panels[0].Series[0].Values.Sum());
        Assert.NotSame(afterRatio, tab.Spec);

        tab.SuspensionConfig.Reset();
        await tab.RunNowAsync();
        Assert.Equal(100.0, tab.Spec!.Panels[0].Series[0].Values.Sum(), 1e-9);
    }

    [Fact]
    public async Task Session_b_gets_its_own_entries_and_a_comparison_chart()
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.Suspension.RunNowAsync();

        var tab = window.Suspension;
        Assert.True(tab.SuspensionConfig.HasSessionB);
        Assert.True(tab.SuspensionConfig.SyncWithA); // same logger and vehicle: B follows A
        Assert.Equal(2, tab.Spec!.Panels[0].Series.Count);
        Assert.StartsWith("Comparison:", tab.Spec.Title);
        Assert.Equal(6, tab.Stats.Count);

        tab.SuspensionConfig.SyncWithA = false;
        tab.SuspensionConfig.ActiveSession = "B";
        Assert.True(tab.SuspensionConfig.Entries[0].IsEditable);
        Assert.Equal("B", tab.SuspensionConfig.ActiveSession);
    }

    [AvaloniaFact]
    public async Task Suspension_view_renders_in_dark_and_light_themes()
    {
        var app = Application.Current!;
        var viewModel = await LoadedWindow();
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        try
        {
            foreach (var (variant, name) in new[] { (ThemeVariant.Dark, "dark"), (ThemeVariant.Light, "light") })
            {
                app.RequestedThemeVariant = variant;
                Avalonia.Threading.Dispatcher.UIThread.RunJobs();
                var frame = window.CaptureRenderedFrame();
                Assert.NotNull(frame);
                frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, $"suspension_{name}.png"), new PngBitmapEncoderOptions());
                Assert.Equal(variant, app.ActualThemeVariant);
                Assert.Equal(ChartSurfaces.ThemeFor(variant), ChartSurfaces.CurrentTheme);
            }
        }
        finally
        {
            app.RequestedThemeVariant = ThemeVariant.Default;
        }
    }
}
