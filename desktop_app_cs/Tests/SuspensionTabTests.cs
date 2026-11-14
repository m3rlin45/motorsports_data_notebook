using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Media.Imaging;
using Avalonia.Styling;
using Avalonia.VisualTree;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.Controls;
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
        Assert.Equal("0.997", tab.SuspensionConfig.RatiosA.FrontLeft.Text);
        Assert.Equal("0.768", tab.SuspensionConfig.RatiosA.RearRight.Text);

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

        tab.SuspensionConfig.RatiosA.FrontLeft.Text = "0.5";
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
        tab.SuspensionConfig.ActiveSession = SessionSlot.B;
        Assert.True(tab.SuspensionConfig.Entries[0].IsEditable);
        Assert.Equal(SessionSlot.B, tab.SuspensionConfig.ActiveSession);
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

    [AvaloniaFact]
    public async Task Vehicle_profile_window_renders_the_suspension_page()
    {
        var viewModel = await LoadedWindow();
        var window = new ProfileWindow(viewModel);
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, "profile_window.png"), new PngBitmapEncoderOptions());

        Assert.Same(viewModel.Suspension.SuspensionConfig, viewModel.SelectedConfigSection!.Config);
        Assert.Contains(window.GetVisualDescendants().OfType<TextBox>(), box => box.Text == "LF_Shock_Pot");
        window.Close();
    }

    [Fact]
    public async Task Session_b_without_laps_is_not_compared_instead_of_blocking_the_tab()
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.SelectNone();

        await window.Suspension.RunNowAsync();

        var tab = window.Suspension;
        Assert.Equal("Analysis complete", tab.Status);
        Assert.Single(tab.Spec!.Panels[0].Series);
        Assert.Equal(3, tab.Stats.Count);
    }

    [Fact]
    public async Task An_invalid_motion_ratio_is_reported_instead_of_silently_replaced()
    {
        var window = await LoadedWindow();
        var tab = window.Suspension;

        tab.SuspensionConfig.RatiosA.FrontLeft.Text = "0,9"; // a decimal comma
        await tab.RunNowAsync();

        Assert.Equal(StatusKind.Hint, tab.StatusKind);
        Assert.Contains("Front-left motion ratio", tab.Status);
        Assert.Contains("decimal point", tab.Status);
        Assert.Null(tab.Spec); // not a chart computed from a default the user never chose
        Assert.Equal(tab.Status, tab.EmptyMessage);

        tab.SuspensionConfig.RatiosA.FrontLeft.Text = "0.9";
        await tab.RunNowAsync();
        Assert.Equal("Analysis complete", tab.Status);
        Assert.NotNull(tab.Spec);
    }

    [Fact]
    public async Task Switching_the_shown_session_does_not_rerun_the_analysis()
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.Suspension.RunNowAsync();
        var ratioChanges = 0;
        window.Suspension.SuspensionConfig.RatiosChanged += () => ratioChanges++;

        window.ProfileSession = SessionSlot.B; // the A/B switch of the profile window
        Assert.Equal(SessionSlot.B, window.Suspension.SuspensionConfig.ActiveSession);
        Assert.Equal(0, ratioChanges);

        window.Suspension.SuspensionConfig.RatiosA.FrontLeft.Text = "0.95";
        Assert.Equal(1, ratioChanges);
        Assert.Equal("0.95", window.Suspension.SuspensionConfig.RatiosB.FrontLeft.Text); // B follows A while synced
    }

    [Fact]
    public async Task Stats_tables_copy_as_tab_separated_text()
    {
        var window = await LoadedWindow();
        var lines = window.Suspension.Stats[0].ToTsv().Split('\n');

        Assert.StartsWith("Statistics", lines[0]);
        Assert.StartsWith("Corner\tSkew\t", lines[1]);
        Assert.Equal(6, lines.Length);
        Assert.StartsWith("FL\t", lines[2]);
    }

    [Fact]
    public async Task Closing_session_b_returns_every_tab_to_single_session_analysis()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        foreach (var tab in window.Tabs)
        {
            tab.Debounce = TimeSpan.Zero;
        }

        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        Assert.True(window.SwapSessionsCommand.CanExecute(null));
        Assert.Contains(" vs ", window.WindowTitle);
        window.ActiveTab = window.Suspension;
        await window.Suspension.RunNowAsync();
        Assert.Equal(2, window.Suspension.Spec!.Panels[0].Series.Count);
        Assert.True(window.Suspension.SuspensionConfig.HasSessionB);

        window.SessionB.Close();
        await window.Suspension.RunNowAsync();

        Assert.Single(window.Suspension.Spec!.Panels[0].Series);
        Assert.False(window.Suspension.SuspensionConfig.HasSessionB);
        Assert.False(window.SwapSessionsCommand.CanExecute(null));
        Assert.DoesNotContain(" vs ", window.WindowTitle);
    }

    [AvaloniaFact]
    public async Task Tab_views_are_kept_across_switches()
    {
        var viewModel = await LoadedWindow();
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var locator = (TabViewLocator)window.FindControl<TabControl>("TabArea")!.ContentTemplate!;
        var view = locator.ViewOf(viewModel.Suspension)!;
        var chart = view.GetLogicalDescendants().OfType<ChartHost>().Single();
        var surface = chart.Surface;
        Assert.NotNull(surface);

        viewModel.ActiveTab = viewModel.TimeDistance;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        viewModel.ActiveTab = viewModel.Suspension;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        // The same view and the same surface: the tab comes back as it was left, with nothing rebuilt.
        Assert.Same(view, locator.ViewOf(viewModel.Suspension));
        Assert.Same(surface, view.GetLogicalDescendants().OfType<ChartHost>().Single().Surface);
        Assert.NotSame(view, locator.ViewOf(viewModel.TimeDistance));
    }

    [Fact]
    public void The_statistics_pane_state_round_trips_through_the_settings_dictionary()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.Suspension.ShowStats = true;
        var state = new Dictionary<string, string>();
        window.Suspension.SaveState(state);
        Assert.Equal("1", state["stats"]);

        var other = new MainWindowViewModel(BuiltinOnly);
        other.Suspension.RestoreState(state);
        Assert.True(other.Suspension.ShowStats);

        // A tab without a statistics pane stores nothing for it.
        state.Clear();
        window.TimeDistance.SaveState(state);
        Assert.DoesNotContain("stats", state.Keys);
    }
}
