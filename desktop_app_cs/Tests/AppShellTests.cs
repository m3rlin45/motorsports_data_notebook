using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.ViewModels;
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

        Assert.Equal("File: stint.ibt | Logger: iracing | Vehicle: FIA F4 | Driver: A Driver", session.FileLabel);
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
        session.SelectionChanged += () => selections++;

        session.Apply(Sample.Value, SessionParityTests.SamplePath());

        Assert.Equal(1, loaded);
        Assert.Equal(1, selections); // the auto-selection, not one per lap
        Assert.Equal("Toyota 86 ZN6", session.Profile?.Name);
        Assert.Contains("Logger: 6701209", session.FileLabel);
        Assert.Equal(Sample.Value.Laps.Count, session.Laps.Count);
        Assert.Equal(LapSelection.GetTopLaps(Sample.Value.Laps).Select(l => l.Num), session.SelectedLaps);
        Assert.Equal("2:05.056", session.Laps.First(l => l.Num == 14).Label.Split(": ")[1]);

        session.SelectBest();
        Assert.Equal(new[] { 14 }, session.SelectedLaps);
        session.SelectNone();
        Assert.Empty(session.SelectedLaps);
        session.SelectAll();
        Assert.Equal(session.Laps.Count, session.SelectedLaps.Count);
        Assert.Equal(4, selections);
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
        Assert.Equal("Analysis complete", window.Status);
        // speed_kmh, PPS, BrakePress plus the delta-time panel.
        Assert.Equal(new[] { "speed_kmh", "PPS", "BrakePress", "Delta time vs reference" }, spec!.Subplots.Select(s => s.Title));
        Assert.Equal(window.SessionA.SelectedLaps.Count, spec.Subplots[0].Traces.Count);
        // The fastest lap is the reference, drawn first.
        Assert.StartsWith("Lap 14", spec.Subplots[0].Traces[0].Label);
        // The lap list is the legend: it carries the trace colours and the plot draws none.
        Assert.False(spec.Legend);
        Assert.Equal(InfernoAnalyzer.App.StripChartBuilder.ColorFor(0), window.SessionA.Laps.First(l => l.Num == 14).TraceColor);
        Assert.Null(window.SessionA.Laps.First(l => l.Num == 1).TraceColor); // not selected, not drawn
        Assert.Equal("GPS Speed", window.TimeDistance.Timebase);
        Assert.Contains(window.TimeDistance.Channels, c => c.Name == "PPS" && c.IsSelected);

        window.TimeDistance.IsTime = true;
        Assert.Equal(3, window.TimeDistance.Spec!.Subplots.Count); // no delta panel against time
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
        Assert.Equal(InfernoAnalyzer.App.StripChartBuilder.ColorFor(0), window.SessionA.Laps.First(l => l.Num == 14).TraceColor);
        var traceB12 = spec.Subplots[0].Traces.Single(t => t.Label.StartsWith("B Lap 12 "));
        Assert.Equal(traceB12.Color, window.SessionB.Laps.First(l => l.Num == 12).TraceColor);
        Assert.Null(window.SessionA.Laps.First(l => l.Num == 1).TraceColor); // not selected, not drawn
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
        Assert.All(window.SessionB.Laps, l => Assert.Null(l.TraceColor));
    }

    [Fact]
    public void Lap_series_builder_translates_channel_names_for_the_comparison_session()
    {
        var log = Sample.Value;
        // Pretend the comparison vehicle logs throttle as "GPS Speed": the PPS panel shows that channel.
        var mapped = InfernoAnalyzer.App.Series.LapSeriesBuilder.Build(log, new[] { 14 }, new[] { "PPS", "BrakePress" }, "GPS Speed", "B", new Dictionary<string, string> { ["PPS"] = "GPS Speed" });
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

        viewModel.ToggleTopPanel();
        Assert.Contains("A: CMD_Inferno", viewModel.CollapsedSummary);
        Assert.Contains("B: No file", viewModel.CollapsedSummary);
    }
}
