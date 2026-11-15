using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
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
    public Task Driver_tab_builds_summary_detail_and_map_with_the_auto_throttle_threshold() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.Driver;

        Assert.StartsWith("Done - ", tab.Status);
        Assert.Contains("TA pts", tab.Status);
        // 95 % of the throttle channel's 95th percentile: PPS tops out near 100.
        var threshold = tab.DriverConfig.ThresholdsA.ThrottleThreshold.Value!.Value;
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
        Assert.False(detail.Legend); // the lap list is the legend: each trace takes its lap's style
        Assert.True(tab.UsesLapColors);
        Assert.Equal(3, detail.Subplots.Count);
        Assert.Equal(window.SessionA.SelectedLaps.Count, detail.Subplots[0].Traces.Count);
        Assert.Contains(detail.Markers!, m => m.Label == "Apex");

        Assert.Equal(2, tab.Stats.Count);
        Assert.Equal(tab.Corners.Count, tab.Stats[0].Rows.Count);
    });

    [Fact]
    public Task The_region_list_has_the_tabs_corners_from_one_run_of_the_kernel() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var regions = window.Regions;
        await regions.LastRefresh;

        // With the default lap ticks the tab and the region list ask for the same analysis: computed once.
        Assert.Equal(1, window.DriverAnalyses.Computations);
        Assert.Equal(window.Driver.Corners.Select(c => c.Index + 1), regions.Corners.Select(c => int.Parse(c.Name["Turn ".Length..])));
        Assert.Matches(@"^Turn 3 · (left|right) · \d+ km/h at the slowest point$", regions.Corners[2].Tip);

        // The tab showing its derived throttle threshold re-lists nothing: same corners, the ticks stay.
        regions.Corners[2].IsSelected = true;
        var changes = 0;
        regions.Changed += () => changes++;
        await regions.RefreshAsync();
        Assert.Equal(0, changes);
        Assert.True(regions.Corners[2].IsSelected);

        // A coarser corner threshold is fewer corners, for the tab and for the list alike. Ticks
        // follow the place, not the number: with every corner ticked, the ones that remain (the
        // tight ones, renumbered) are all still ticked.
        foreach (var corner in regions.Corners)
        {
            corner.IsSelected = true;
        }

        var thresholds = window.Driver.DriverConfig.ThresholdsA;
        var before = regions.Corners.Count;
        changes = 0;
        thresholds.CornerThreshold.Text = "0.012";
        await window.Driver.RunNowAsync();
        await regions.LastRefresh;
        Assert.InRange(regions.Corners.Count, 1, before - 1);
        Assert.Equal(window.Driver.Corners.Count, regions.Corners.Count);
        Assert.True(changes > 0); // fewer regions: the tabs recompute
        Assert.All(regions.Corners, c => Assert.True(c.IsSelected));

        // An invalid threshold lists no corners and says why.
        thresholds.CornerThreshold.Text = "abc";
        await regions.LastRefresh;
        Assert.Empty(regions.Corners);
        Assert.Contains("threshold is not valid", regions.Hint);
    });

    [Fact]
    public Task Box_statistics_follow_numpy_and_matplotlib_conventions() => Ui.Run(() =>
    {
        var box = BoxSpec.From(1, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 }, "#000000", "tip");

        Assert.Equal(3.25, box.Q1);
        Assert.Equal(5.5, box.Median);
        Assert.Equal(7.75, box.Q3);
        Assert.Equal(1, box.WhiskerLow);
        Assert.Equal(9, box.WhiskerHigh);
        Assert.Equal(new double[] { 100 }, box.Fliers);
        Assert.Equal(14.5, box.Mean);
        Assert.Contains("Median 5.50", DriverChartBuilder.BoxStatsText(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 100 }, "Turn 1 (L)", "%"));
    });

    [Fact]
    public Task Driver_views_render() => Ui.Run(async () =>
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
            RenderSmokeTests.SaveRender(frame!, $"driver_{name}.png");
        }
    });

    [Fact]
    public Task Session_b_without_laps_is_not_compared() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        window.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        window.SessionB.SelectNone();

        await window.Driver.RunNowAsync();

        Assert.StartsWith("Done - ", window.Driver.Status);
        Assert.StartsWith("Driver Consistency:", window.Driver.SummarySpec!.Title);
        Assert.Empty(window.Driver.SummarySpec.Legend);
    });

    [Fact]
    public Task Opening_a_corner_shows_its_detail_even_when_it_is_already_selected() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var tab = window.Driver;
        Assert.Equal(tab.Corners[0], tab.SelectedCorner); // preselected after the analysis
        Assert.Equal(DriverViewMode.Summary, tab.ViewMode);

        tab.OpenCorner(0);
        Assert.Equal(DriverViewMode.Detail, tab.ViewMode);
        var key = tab.DetailSpec!.ViewKey;

        tab.ShowSummary(); // Esc
        Assert.Equal(DriverViewMode.Summary, tab.ViewMode);
        tab.OpenCorner(3);
        Assert.Equal(tab.Corners[3], tab.SelectedCorner);
        Assert.Equal(DriverViewMode.Detail, tab.ViewMode);
        Assert.NotEqual(key, tab.DetailSpec!.ViewKey); // another corner starts from its own view

        tab.OpenCorner(99); // out of range: ignored
        Assert.Equal(tab.Corners[3], tab.SelectedCorner);

        // Each lap's trace carries the style its lap-list row shows.
        var lap14 = tab.DetailSpec.Subplots[0].Traces.Single(t => t.Label == "Lap 14");
        Assert.Equal(window.SessionA.StyleOf(14)!.Value.Color, lap14.Color);
        Assert.Equal("L14", lap14.ShortLabel);

        // Every map apex names its corner, so a click on it can open that corner.
        var ids = tab.MapSpec!.Points.Where(p => p.Id >= 0).Select(p => p.Id).ToList();
        Assert.Equal(Enumerable.Range(0, tab.Corners.Count), ids.OrderBy(i => i));
    });

    [Fact]
    public Task Clicking_a_corner_in_the_summary_or_on_the_map_opens_it() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var tab = viewModel.Driver;
        var view = new DriverView { DataContext = tab };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        window.CaptureRenderedFrame();

        var summary = (InfernoAnalyzer.Charting.ScottPlot.ScottPlotBoxPlot)view.FindControl<InfernoAnalyzer.App.Controls.ChartHost>("SummaryChart")!.Surface!;
        Assert.Equal(2, summary.CategoryAt(tab.SummarySpec!.CategoryPositions[2] + 0.3));
        Assert.Equal(-1, summary.CategoryAt(tab.SummarySpec.CategoryPositions[^1] + 2));

        tab.ViewMode = DriverViewMode.Map;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        window.CaptureRenderedFrame();
        var map = (InfernoAnalyzer.Charting.ScottPlot.ScottPlotTrackMap)view.FindControl<InfernoAnalyzer.App.Controls.ChartHost>("MapChart")!.Surface!;
        var apex = tab.MapSpec!.Points.First(p => p.Id == 4);
        Assert.Equal(4, map.PointNear(new global::ScottPlot.Coordinates(apex.X, apex.Y)));
        Assert.Equal(-1, map.PointNear(new global::ScottPlot.Coordinates(apex.X + 5000, apex.Y)));
    });

    [Fact]
    public Task The_small_map_shows_the_corner_under_the_pointer_in_the_box_plots() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var tab = viewModel.Driver;
        var view = new DriverView { DataContext = tab };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        window.CaptureRenderedFrame();

        // One highlight per corner, by index, on a compact map with a clickable dot at each apex.
        var spec = tab.MiniMapSpec!;
        Assert.True(spec.Compact);
        Assert.Equal(Enumerable.Range(0, tab.Corners.Count), spec.Highlights!.Select(h => h.Id));
        Assert.Equal(Enumerable.Range(0, tab.Corners.Count), spec.Points.Select(p => p.Id));
        Assert.All(spec.Highlights!, h => Assert.True(h.X.Length > 1));

        var map = (InfernoAnalyzer.Charting.ScottPlot.ScottPlotTrackMap)view.FindControl<InfernoAnalyzer.App.Controls.ChartHost>("MiniMap")!.Surface!;
        var summaryHost = view.FindControl<InfernoAnalyzer.App.Controls.ChartHost>("SummaryChart")!;
        var summary = (InfernoAnalyzer.Charting.ScottPlot.ScottPlotBoxPlot)summaryHost.Surface!;
        var firstPanel = summaryHost.GetVisualDescendants().OfType<global::ScottPlot.Avalonia.AvaPlot>().First();

        // Nothing under the pointer: the corner selected in the list.
        Assert.Equal(-1, tab.HoveredCorner);
        Assert.Equal(tab.SelectedCorner!.Index, map.HighlightedItem);
        Assert.True(map.VisibleHighlightParts > 0);

        Point OverColumn(int corner)
        {
            var limits = summary.Plots[0].Axes.GetLimits();
            var pixel = summary.Plots[0].GetPixel(new global::ScottPlot.Coordinates(tab.SummarySpec!.CategoryPositions[corner], (limits.Bottom + limits.Top) / 2));
            return firstPanel.TranslatePoint(new Point(pixel.X, pixel.Y), window)!.Value;
        }

        window.MouseMove(OverColumn(4));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(4, tab.HoveredCorner);
        Assert.Equal(4, map.HighlightedItem);

        window.MouseMove(OverColumn(2));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(2, map.HighlightedItem);
        var frame = window.CaptureRenderedFrame();
        RenderSmokeTests.SaveRender(frame!, "driver_minimap_hover.png");

        // Off the chart again: back to the selected corner.
        var list = view.FindControl<ListBox>("CornerList")!;
        window.MouseMove(list.TranslatePoint(new Point(20, list.Bounds.Height - 10), window)!.Value);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(-1, tab.HoveredCorner);
        Assert.Equal(tab.SelectedCorner.Index, map.HighlightedItem);

        // The selection moves it too, and a click on the small map opens that corner.
        tab.SelectedCorner = tab.Corners[5];
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(5, map.HighlightedItem);
        var apex = spec.Points.First(p => p.Id == 3);
        Assert.Equal(3, map.PointNear(new global::ScottPlot.Coordinates(apex.X, apex.Y)));
    });

    [Fact]
    public Task Each_tab_shows_its_own_lap_selection_and_changes_reach_only_the_tabs_that_use_it() => Ui.Run(() =>
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        var top103 = LapSelection.GetTopLaps(Sample.Value.Laps).Select(l => l.Num).ToList();

        // Time/Distance is the startup tab: the best two laps, with their colour keys.
        Assert.Equal(new[] { 12, 14 }, window.SessionA.SelectedLaps);
        Assert.NotNull(window.SessionA.Laps.First(l => l.Num == 14).Style);

        window.ActiveTab = window.Suspension;
        Assert.Equal(top103, window.SessionA.SelectedLaps);
        Assert.All(window.SessionA.Laps, l => Assert.Null(l.Style)); // no per-lap traces on this tab

        // Unticking a lap here is a change to the statistical selection only.
        window.SessionA.Laps.First(l => l.Num == 14).IsSelected = false;
        Assert.DoesNotContain(14, window.SessionA.SelectedIn(LapScope.Analysis));
        Assert.Contains(14, window.SessionA.SelectedIn(LapScope.Traces));

        window.ActiveTab = window.TimeDistance;
        Assert.Equal(new[] { 12, 14 }, window.SessionA.SelectedLaps);

        // The Driver tab's corner detail draws a trace per lap, so the keys come back with it.
        window.ActiveTab = window.Driver;
        Assert.All(window.SessionA.Laps, l => Assert.Null(l.Style));
        window.Driver.ViewMode = DriverViewMode.Detail;
        Assert.NotNull(window.SessionA.Laps.First(l => l.Num == 12).Style);
    });
}
