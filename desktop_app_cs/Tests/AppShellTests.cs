using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.Controls;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The session slot and window view models over the Fuji 86 sample.</summary>
public class AppShellTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");

    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    /// <summary>The label of an iRacing session comes from the metadata libibt derives from the session info.</summary>
    [Fact]
    public void Session_label_names_the_iracing_car_and_driver()
    {
        var speed = new Channel("Speed", new long[] { 0, 1000, 2000 }, new double[] { 30, 40, 50 });
        var metadata = new Dictionary<string, string>
        {
            ["session_info_yaml"] = "---\nWeekendInfo:\n TrackName: phillipisland\n",
            ["car_name"] = "FIA F4",
            ["driver_name"] = "A Driver",
        };
        var log = new LogFile(new[] { speed }, new List<Lap> { new(1, 0, 2000, "full") }, metadata, "stint.ibt");
        var session = new SessionViewModel("A", BuiltinOnly);

        session.Apply(log, "/tmp/stint.ibt");

        Assert.Equal("stint.ibt", session.FileName);
        Assert.Equal("Logger iracing · FIA F4 · A Driver", session.Details);
        Assert.Equal("iRacing", session.Profile?.Name);
        Assert.Null(session.VehicleName); // the profile lookup is by logger ID alone, as in Python
    }

    [Fact]
    public void Session_slot_lists_laps_resolves_the_profile_and_preselects_the_top_laps()
    {
        var session = new SessionViewModel("SESSION A (Primary)", BuiltinOnly);
        var loaded = 0;
        var selections = 0;
        session.SessionLoaded += () => loaded++;
        session.SelectionChanged += _ => selections++;

        session.Apply(Sample.Value, SessionParityTests.SamplePath());

        Assert.Equal(1, loaded);
        Assert.Equal(2, selections); // the auto-selection of each scope, not one per lap
        Assert.Equal("Toyota 86 ZN6", session.Profile?.Name);
        Assert.Contains("Logger 6701209", session.Details);
        Assert.Equal("Profile: Toyota 86 ZN6", session.ProfileName);
        Assert.Equal(Sample.Value.Laps.Count, session.Laps.Count);
        Assert.Equal(LapSelection.GetTopLaps(Sample.Value.Laps).Select(l => l.Num), session.SelectedLaps);
        // The trace overlay starts from the best two laps; the statistical tabs from the top 103 %.
        Assert.Equal(new[] { 12, 14 }, session.SelectedIn(LapScope.Traces));
        var best = session.Laps.First(l => l.Num == 14);
        Assert.Equal("2:05.056", best.TimeText);
        Assert.True(best.IsBest);
        Assert.Equal("best", best.GapText);
        Assert.StartsWith("+", session.Laps.First(l => l.Num == 12).GapText);

        session.SelectBest();
        Assert.Equal(new[] { 14 }, session.SelectedLaps);
        session.SelectNone();
        Assert.Empty(session.SelectedLaps);
        session.SelectAll();
        Assert.Equal(session.Laps.Count, session.SelectedLaps.Count);
        Assert.Equal(5, selections);
    }

    [Fact]
    public async Task Loading_a_bad_file_reports_the_error_without_a_session()
    {
        var session = new SessionViewModel("A", BuiltinOnly);
        var bad = Path.Combine(Path.GetTempPath(), $"inferno-bad-{Guid.NewGuid():N}.xrz");
        await File.WriteAllBytesAsync(bad, new byte[] { 1, 2, 3 }, TestContext.Current.CancellationToken);
        try
        {
            await session.LoadFileAsync(bad);
        }
        finally
        {
            File.Delete(bad);
        }

        Assert.False(session.HasSession);
        Assert.Empty(session.Laps);
    }

    [Fact]
    public async Task Time_distance_tab_builds_a_strip_chart_from_the_selected_laps()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        var completed = 0;
        window.TimeDistance.AnalysisCompleted += () => completed++;

        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TimeDistance.RunNowAsync();

        var spec = window.TimeDistance.Spec;
        Assert.NotNull(spec);
        // With a zero debounce the session-load and selection triggers cannot coalesce, so
        // the load's analysis runs and the selection's rerun follows it.
        Assert.InRange(completed, 1, 2);
        Assert.Equal("Analysis complete", window.ActiveTab.Status);
        // speed_kmh, PPS, BrakePress plus the delta-time panel.
        Assert.Equal(new[] { "Speed", "PPS", "BrakePress", "Delta time vs reference" }, spec!.Subplots.Select(s => s.Title));
        Assert.Equal(window.SessionA.SelectedLaps.Count, spec.Subplots[0].Traces.Count);
        // The fastest lap is the reference, drawn first.
        Assert.StartsWith("Lap 14", spec.Subplots[0].Traces[0].Label);
        // The lap list is the legend: it carries the trace colours and the plot draws none.
        Assert.False(spec.Legend);
        Assert.Equal(new LapStyle(0), window.SessionA.Laps.First(l => l.Num == 14).Style);
        Assert.Equal(ChartColors.Lap(0), spec.Subplots[0].Traces[0].Color);
        Assert.True(spec.Subplots[0].Traces[0].Reference);
        Assert.Null(window.SessionA.Laps.First(l => l.Num == 1).Style); // not selected, not drawn
        Assert.Equal("GPS Speed", window.TimeDistance.Timebase);
        Assert.Contains(window.TimeDistance.Channels, c => c.Name == "PPS" && c.IsSelected);

        window.TimeDistance.IsTime = true;
        Assert.Equal(3, window.TimeDistance.Spec!.Subplots.Count); // no delta panel against time

        // The channel search hides non-matching rows but keeps their selection.
        window.TimeDistance.ChannelFilter = "shock";
        Assert.All(window.TimeDistance.Channels, c => Assert.Equal(c.Name.Contains("shock", StringComparison.OrdinalIgnoreCase), c.IsMatch));
        Assert.Contains(window.TimeDistance.Channels, c => c.IsMatch);
        Assert.Contains(window.TimeDistance.Channels, c => c.Name == "PPS" && c.IsSelected && !c.IsMatch);
        window.TimeDistance.ChannelFilter = string.Empty;
        Assert.All(window.TimeDistance.Channels, c => Assert.True(c.IsMatch));
    }

    [Fact]
    public async Task Time_distance_tab_overlays_session_b_laps_with_session_labels()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TimeDistance.RunNowAsync();

        var spec = window.TimeDistance.Spec!;
        Assert.False(spec.Legend); // the lap lists are the legend
        var countA = window.SessionA.SelectedLaps.Count;
        var countB = window.SessionB.SelectedLaps.Count;
        Assert.True(countB > 0);
        Assert.Equal(countA + countB, spec.Subplots[0].Traces.Count);
        Assert.Equal(countA, spec.Subplots[0].Traces.Count(t => t.Label.StartsWith("A Lap ")));
        Assert.Equal(countB, spec.Subplots[0].Traces.Count(t => t.Label.StartsWith("B Lap ")));
        // Fastest lap of either session is the delta reference.
        Assert.StartsWith("A Lap 14", spec.Subplots[0].Traces[0].Label);
        // The lap lists carry the trace colours (they are the legend).
        Assert.Equal(new LapStyle(0), window.SessionA.Laps.First(l => l.Num == 14).Style);
        var traceB12 = spec.Subplots[0].Traces.Single(t => t.Label.StartsWith("B Lap 12 "));
        Assert.Equal(traceB12.Color, window.SessionB.Laps.First(l => l.Num == 12).Style!.Value.Color);
        // Four laps on screen, four different styles.
        Assert.Equal(4, spec.Subplots[0].Traces.Select(t => (t.Color, t.Dashed)).Distinct().Count());
        Assert.Null(window.SessionA.Laps.First(l => l.Num == 1).Style); // not selected, not drawn
        Assert.Contains("vs A Lap 14", spec.Subplots[^1].Traces[1].Label);
        Assert.StartsWith($"{countA + countB} laps", window.TimeDistance.Summary);

        // Same vehicle on both sides: nothing to translate.
        Assert.Empty(window.TimeDistance.ChannelMapToB());

        // Deselecting every B lap falls back to A alone.
        foreach (var lap in window.SessionB.Laps)
        {
            lap.IsSelected = false;
        }

        await window.TimeDistance.RunNowAsync();
        Assert.Equal(countA, window.TimeDistance.Spec!.Subplots[0].Traces.Count);
        Assert.StartsWith("Lap 14", window.TimeDistance.Spec.Subplots[0].Traces[0].Label);
        Assert.All(window.SessionB.Laps, l => Assert.Null(l.Style));
    }

    [Fact]
    public void Lap_series_builder_translates_channel_names_for_the_comparison_session()
    {
        var log = Sample.Value;
        // Pretend the comparison vehicle logs throttle as "GPS Speed": the PPS panel shows that channel.
        var mapped = InfernoAnalyzer.App.Series.LapSeriesBuilder.Build(log, new[] { 14 }, new[] { "PPS", "BrakePress" }, "GPS Speed", "B", new Dictionary<string, string> { ["PPS"] = "GPS Speed" }, TestContext.Current.CancellationToken);
        var plain = InfernoAnalyzer.App.Series.LapSeriesBuilder.Build(log, new[] { 14 }, new[] { "GPS Speed", "BrakePress" }, "GPS Speed");

        var lap = Assert.Single(mapped.Laps);
        Assert.Equal("B", lap.Session);
        Assert.Equal("B Lap 14", lap.ShortLabel);
        Assert.Equal(plain.Laps[0].Series["GPS Speed"], lap.Series["PPS"]);
        Assert.Equal(plain.Laps[0].Series["BrakePress"], lap.Series["BrakePress"]);
        Assert.Equal(plain.Units["GPS Speed"], mapped.Units["PPS"]);
        Assert.NotNull(lap.Before);
        Assert.Equal("B", lap.Before!.Session);
    }

    [AvaloniaFact]
    public async Task Main_window_renders_with_a_loaded_session()
    {
        var viewModel = new MainWindowViewModel(BuiltinOnly);
        viewModel.TimeDistance.Debounce = TimeSpan.Zero;
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();

        viewModel.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await viewModel.TimeDistance.RunNowAsync();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        viewModel.TimeDistance.Cursors.SetCursor(1500);

        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, "main_window_session.png"), new PngBitmapEncoderOptions());

        // The status line shows the startup tab's status without a tab switch first.
        Assert.Equal("Analysis complete", window.FindControl<TextBlock>("StatusText")!.Text);
        Assert.StartsWith("CMD_Inferno", viewModel.WindowTitle);
        Assert.EndsWith("Inferno Analyzer", window.Title);

        viewModel.ToggleSidebar();
        Assert.True(viewModel.IsSidebarCollapsed);
    }

    [Fact]
    public void Closing_a_session_unloads_it_and_tells_the_tabs()
    {
        var session = new SessionViewModel("A", BuiltinOnly);
        var loaded = 0;
        session.SessionLoaded += () => loaded++;
        Assert.False(session.CloseCommand.CanExecute(null));

        session.Apply(Sample.Value, SessionParityTests.SamplePath());
        Assert.True(session.CloseCommand.CanExecute(null));
        session.Close();

        Assert.Equal(2, loaded);
        Assert.False(session.HasSession);
        Assert.Null(session.Log);
        Assert.Empty(session.Laps);
        Assert.Empty(session.SelectedIn(LapScope.Analysis));
        Assert.Equal(string.Empty, session.FileName);
        Assert.Equal("Session", session.SessionLabel);
    }

    [Fact]
    public async Task Ticked_channels_survive_the_next_session_and_a_comparison_load()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        var tab = window.TimeDistance;
        Assert.Equal(new[] { "speed_kmh", "PPS", "BrakePress" }, tab.ShownChannels.Select(c => c.Name));

        tab.Channels.First(c => c.Name == "LF_Shock_Pot").IsSelected = true;
        tab.Channels.First(c => c.Name == "BrakePress").IsSelected = false;
        tab.MoveShown(2, 0); // the shock pot to the top panel
        var picked = new[] { "LF_Shock_Pot", "speed_kmh", "PPS" };
        Assert.Equal(picked, tab.ShownChannels.Select(c => c.Name));

        // Loading a comparison session must not reset what is plotted.
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        Assert.Equal(picked, tab.ShownChannels.Select(c => c.Name));

        // Nor must the next run of the same car.
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        Assert.Equal(picked, tab.ShownChannels.Select(c => c.Name));
        Assert.True(tab.Channels.First(c => c.Name == "LF_Shock_Pot").IsSelected);

        await tab.RunNowAsync();
        Assert.Equal("LF_Shock_Pot", tab.Spec!.Subplots[0].Title); // panel order follows the list

        // The set is remembered per vehicle for the next launch.
        Assert.Equal(picked, window.Settings.ChannelsByVehicle["6701209|Inferno 86"]);
    }

    [Fact]
    public async Task Search_enter_ticks_the_first_match()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        var tab = window.TimeDistance;
        var asked = 0;
        tab.SearchRequested += () => asked++;
        tab.FindChannelCommand.Execute(null);
        Assert.Equal(1, asked);

        tab.ToggleFirstMatch(); // no search: nothing happens
        Assert.Equal(3, tab.ShownChannels.Count);

        tab.ChannelFilter = "lf_shock";
        Assert.Equal("LF_Shock_Pot", tab.FilteredChannels[0].Name);
        tab.ToggleFirstMatch();
        Assert.Contains(tab.ShownChannels, c => c.Name == "LF_Shock_Pot");
        tab.ToggleFirstMatch();
        Assert.DoesNotContain(tab.ShownChannels, c => c.Name == "LF_Shock_Pot");
        await Task.CompletedTask;
    }

    [Fact]
    public async Task Redrawing_for_the_same_session_keeps_the_view_key_and_a_new_file_changes_it()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.TimeDistance.RunNowAsync();
        var key = window.TimeDistance.Spec!.ViewKey;
        Assert.NotNull(key);

        window.SessionA.Laps.First(l => l.Num == 10).IsSelected = true;
        await window.TimeDistance.RunNowAsync();
        Assert.Equal(3, window.TimeDistance.Spec!.Subplots[0].Traces.Count);
        Assert.Equal(key, window.TimeDistance.Spec.ViewKey); // the chart keeps its zoom

        window.SessionA.Apply(Sample.Value, "/elsewhere/other_run.xrz");
        await window.TimeDistance.RunNowAsync();
        Assert.NotEqual(key, window.TimeDistance.Spec!.ViewKey);
    }

    [Fact]
    public async Task Opening_two_files_fills_both_sessions_and_one_goes_to_the_target()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        var sample = SessionParityTests.SamplePath();
        Assert.True(MainWindowViewModel.IsTelemetryFile("run.XRK"));
        Assert.False(MainWindowViewModel.IsTelemetryFile("notes.txt"));

        await window.OpenFilesAsync(new[] { "notes.txt", sample }, SessionSlot.B);
        Assert.False(window.SessionA.HasSession);
        Assert.True(window.SessionB.HasSession);

        await window.OpenFilesAsync(new[] { sample, sample, sample });
        Assert.True(window.SessionA.HasSession);
        Assert.True(window.SessionB.HasSession);
        Assert.Equal(new[] { sample }, window.RecentFiles);
    }

    [Fact]
    public async Task Settings_remember_recent_files_and_view_state_between_runs()
    {
        var dir = Directory.CreateTempSubdirectory("inferno-settings");
        try
        {
            var path = Path.Combine(dir.FullName, "nested", "analyzer_settings.json");
            var first = new MainWindowViewModel(BuiltinOnly, settings: new SettingsStore(path));
            await first.OpenFilesAsync(new[] { SessionParityTests.SamplePath() });
            first.TimeDistance.XAxis = XAxisMode.Time;
            first.IsSidebarCollapsed = true;
            first.Theme = "Dark";
            first.Settings.WindowWidth = 1500;
            first.SaveSettings();

            var second = new MainWindowViewModel(BuiltinOnly, settings: new SettingsStore(path));
            Assert.Equal(XAxisMode.Time, second.TimeDistance.XAxis);
            Assert.True(second.IsSidebarCollapsed);
            Assert.Equal("Dark", second.Theme);
            Assert.Equal(1500, second.Settings.WindowWidth);
            Assert.Equal(new[] { SessionParityTests.SamplePath() }, second.RecentFiles);
            Assert.Equal(Path.GetDirectoryName(SessionParityTests.SamplePath()), second.Settings.LastFolder);

            // A damaged settings file is a fresh start, not a failure to launch.
            await File.WriteAllTextAsync(path, "{ not json", TestContext.Current.CancellationToken);
            var third = new MainWindowViewModel(BuiltinOnly, settings: new SettingsStore(path));
            Assert.Equal(XAxisMode.Distance, third.TimeDistance.XAxis);
            Assert.Empty(third.RecentFiles);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Recent_files_are_newest_first_without_duplicates_and_capped()
    {
        var settings = new AppSettings();
        for (var i = 0; i < AppSettings.MaxRecentFiles + 3; i++)
        {
            settings.AddRecent($"/data/run{i}.xrk");
        }

        settings.AddRecent("/data/run5.xrk");

        Assert.Equal(AppSettings.MaxRecentFiles, settings.RecentFiles.Count);
        Assert.Equal("/data/run5.xrk", settings.RecentFiles[0]);
        Assert.Single(settings.RecentFiles, p => p == "/data/run5.xrk");
        Assert.Equal("/data", settings.LastFolder);
    }

    [AvaloniaFact]
    public async Task Charts_show_an_empty_state_until_there_is_data_and_again_when_there_is_none()
    {
        var viewModel = new MainWindowViewModel(BuiltinOnly);
        viewModel.TimeDistance.Debounce = TimeSpan.Zero;
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var locator = (InfernoAnalyzer.App.Views.TabViewLocator)window.FindControl<TabControl>("TabArea")!.ContentTemplate!;
        var chart = locator.ViewOf(viewModel.TimeDistance)!.GetLogicalDescendants().OfType<ChartHost>().Single();
        Assert.Null(chart.Surface); // nothing to draw yet
        Assert.Equal(AnalysisTabViewModel.NoSessionHint, chart.EmptyText);
        Assert.Equal(AnalysisTabViewModel.NoSessionHint, window.FindControl<TextBlock>("StatusText")!.Text);
        Assert.Null(chart.RenderImage());
        var empty = window.CaptureRenderedFrame();
        empty!.Save(Path.Combine(RenderSmokeTests.RenderDir, "main_window_empty.png"), new PngBitmapEncoderOptions());

        viewModel.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await viewModel.TimeDistance.RunNowAsync();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        var surface = chart.Surface;
        Assert.NotNull(surface);
        Assert.True(surface!.Control.IsVisible);
        using var image = chart.RenderImage(); // what Copy image puts on the clipboard
        Assert.NotNull(image);

        // No laps: the stale chart is hidden behind the reason, not left on screen.
        viewModel.SessionA.SelectNone();
        await viewModel.TimeDistance.RunNowAsync();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.False(surface.Control.IsVisible);
        Assert.Equal("Select laps to overlay", chart.EmptyText);
    }
}
