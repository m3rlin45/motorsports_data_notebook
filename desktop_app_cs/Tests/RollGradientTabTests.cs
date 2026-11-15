using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Charting.ScottPlot;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The Roll Gradient tab over the Fuji 86 sample.</summary>
public class RollGradientTabTests
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

        window.ActiveTab = window.RollGradient;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.RollGradient.RunNowAsync();
        return window;
    }

    [Fact]
    public Task Roll_gradient_tab_builds_the_map_and_three_panels_from_the_profile() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.RollGradient;

        Assert.Equal("Analysis complete", tab.Status);
        Assert.Equal("1520", tab.RollConfig.WidthsA.Front.Text);
        Assert.Equal("1540", tab.RollConfig.WidthsA.Rear.Text);
        var spec = tab.Spec!;
        Assert.Single(spec.Paths);
        Assert.Equal(3, spec.Panels.Count);
        Assert.Equal(spec.Paths[0].X.Length, spec.Panels[0].Scatter[0].X.Length); // index-aligned
        Assert.InRange(spec.Paths[0].X.Length, 1000, 24000); // downsampled with step = n / 12000, as in Python
        Assert.Single(spec.Panels[0].Lines);
        Assert.StartsWith("gradient =", spec.Panels[0].Annotation);
        Assert.StartsWith("Roll Gradient  |  front", spec.Title);
        Assert.Single(tab.Stats);
        Assert.Equal(3, tab.Stats[0].Rows.Count);
    });

    [Fact]
    public Task Missing_track_widths_block_the_analysis_with_a_message() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.RollGradient;

        tab.RollConfig.WidthsA.Front.Text = string.Empty;
        await tab.RunNowAsync();

        Assert.Equal("Enter the front and rear track widths (mm) under Channels", tab.Status);
    });

    [Fact]
    public Task Hovering_the_track_map_highlights_samples_and_the_view_renders() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var view = new RollGradientView { DataContext = viewModel.RollGradient };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        window.CaptureRenderedFrame();

        var surface = (ScottPlotScatterGrid)view.Chart.Surface!;
        var path = viewModel.RollGradient.Spec!.Paths[0];
        surface.HighlightNear(path.X[100], path.Y[100]);
        Assert.True(surface.HighlightedCount > 0);
        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        RenderSmokeTests.SaveRender(frame!, "roll_gradient.png");

        surface.ClearHighlights();
        Assert.Equal(0, surface.HighlightedCount);

        // A real pointer move far from the track clears; near it highlights.
        window.MouseMove(new Point(5, 5));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(0, surface.HighlightedCount);
    });

    [Fact]
    public Task A_ticked_region_restricts_the_roll_gradient_with_the_whole_lap_fit_for_reference() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        await viewModel.Regions.LastRefresh; // started by the session load
        var tab = viewModel.RollGradient;
        viewModel.ActiveTab = tab;
        await tab.RunNowAsync();
        var wholeNote = tab.Spec!.Panels[0].Annotation; // gradient, R² and the sample count
        var wholeLines = tab.Spec.Panels[0].Lines.Count;
        var wholeTitle = tab.Spec.Title;

        viewModel.Regions.Corners[2].IsSelected = true;
        await tab.RunNowAsync();

        var spec = tab.Spec!;
        Assert.EndsWith("  |  Turn 3", spec.Title);
        Assert.DoesNotContain(wholeNote, spec.Panels[0].Annotation); // a fit of the corner's samples only
        Assert.Equal(wholeLines + 1, spec.Panels[0].Lines.Count); // its own fit and the whole laps' in grey
        Assert.Equal(InfernoAnalyzer.Charting.ChartColors.Neutral, spec.Panels[0].Lines[^1].Color);
        Assert.Contains("whole lap", spec.Panels[0].Annotation);
        Assert.Contains("Turn 3", tab.Stats[0].Title);

        var view = new RollGradientView { DataContext = tab };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "roll_gradient_region.png");

        viewModel.Regions.ShowWholeLap = false;
        await tab.RunNowAsync();
        Assert.Equal(wholeLines, tab.Spec!.Panels[0].Lines.Count);

        // Two sessions are compared with each other: no whole-lap line, although the reference is still ticked.
        viewModel.Regions.ShowWholeLap = true;
        viewModel.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await tab.RunNowAsync();
        Assert.EndsWith("  |  Turn 3", tab.Spec!.Title);
        Assert.Equal(2, tab.Spec.Panels[0].Scatter.Count);
        Assert.DoesNotContain(tab.Spec.Panels[0].Lines, l => l.Color == InfernoAnalyzer.Charting.ChartColors.Neutral);
        Assert.DoesNotContain("whole lap", tab.Spec.Panels[0].Annotation);
        viewModel.SessionB.Close();

        viewModel.Regions.Clear();
        await tab.RunNowAsync();
        Assert.Equal(wholeTitle, tab.Spec!.Title);
        Assert.Equal(wholeNote, tab.Spec.Panels[0].Annotation);
    });

    [Fact]
    public Task Session_b_without_laps_is_not_compared_and_min_g_is_validated() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.RollGradient;
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.SelectNone();
        await tab.RunNowAsync();
        Assert.Equal("Analysis complete", tab.Status);
        Assert.Single(tab.Spec!.Paths);

        // The profile window's A/B switch moves every page together.
        window.ProfileSession = SessionSlot.B;
        Assert.Equal(SessionSlot.B, tab.RollConfig.ActiveSession);
        Assert.Equal(SessionSlot.B, window.Suspension.SuspensionConfig.ActiveSession);
        window.ProfileSession = SessionSlot.A;

        tab.MinLateralG.Text = "-1";
        await tab.RunNowAsync();
        Assert.Contains("Min |G| for fit must be at least 0", tab.Status);
        Assert.Null(tab.Spec);

        tab.MinLateralG.Text = "0.3";
        await tab.RunNowAsync();
        Assert.Equal("Analysis complete", tab.Status);
        var state = new Dictionary<string, string>();
        tab.SaveState(state);
        Assert.Equal("0.3", state["min_g"]);
    });
}
