using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.VisualTree;
using InfernoAnalyzer.App;
using InfernoAnalyzer.App.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.Tests;

/// <summary>The sidebar's region selector: one choice of corners and sectors for every tab.</summary>
public class RegionSelectorTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly TrackStore NoTracks = new("/nonexistent/tracks.yaml"); // not this machine's saved tracks
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static async Task<MainWindowViewModel> LoadedWindow(AnalysisTabViewModel? active = null)
    {
        var window = new MainWindowViewModel(BuiltinOnly, NoTracks);
        foreach (var tab in window.Tabs)
        {
            tab.Debounce = TimeSpan.Zero;
        }

        window.ActiveTab = active is null ? window.Suspension : window.Tabs.First(t => t.Header == active.Header);
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.ActiveTab.RunNowAsync();
        await window.Regions.LastRefresh;
        return window;
    }

    [Fact]
    public Task The_selector_appears_with_a_session_and_says_what_the_ticks_do_on_each_tab() => Ui.Run(async () =>
    {
        var empty = new MainWindowViewModel(BuiltinOnly, NoTracks);
        Assert.False(empty.Regions.HasSession); // nothing to pick regions of: the card is not shown

        var window = await LoadedWindow();
        var regions = window.Regions;
        Assert.True(regions.HasSession);
        Assert.Equal(new[] { "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9" }, regions.Corners.Select(c => c.ShortName));
        Assert.Equal("Turn 3", regions.Corners[2].Name);

        // The statistical tabs cover only the ticked regions, the Driver tab lists their corners
        // and Time/Distance brings them into view; Sectors shows whole laps and the selector says so.
        foreach (var tab in window.Tabs)
        {
            window.ActiveTab = tab;
            var uses = tab is not SectorsViewModel; // Sectors is where sectors are defined, not a view of them
            Assert.Equal(uses, regions.IsUsed);
            Assert.Equal(uses ? string.Empty : "Not used on this tab: it shows whole laps", regions.Caption);
        }

        // The ticks are the session's, not a tab's: they wait through a tab that does not use them.
        window.ActiveTab = window.Suspension;
        regions.Corners[2].IsSelected = true;
        window.ActiveTab = window.Sectors;
        window.ActiveTab = window.TireGrip;
        await window.TireGrip.RunNowAsync();
        Assert.True(regions.Corners[2].IsSelected);
        Assert.Equal("Turn 3", window.TireGrip.RegionLabel);

        // The tag's cross on a tab is the same "whole laps" as the selector's.
        regions.ClearCommand.Execute(null);
        await window.TireGrip.RunNowAsync();
        Assert.False(window.TireGrip.IsRegional);
        Assert.False(regions.HasSelection);

        window.SessionA.Close();
        await regions.LastRefresh;
        Assert.False(regions.HasSession);
    });

    [Fact]
    public Task A_double_click_ticks_one_region_alone() => Ui.Run(async () =>
    {
        var window = await LoadedWindow();
        var regions = window.Regions;
        regions.Corners[0].IsSelected = true;
        regions.Corners[4].IsSelected = true;
        var changes = 0;
        regions.Changed += () => changes++;

        regions.Solo(6);
        Assert.Equal(new[] { "Turn 7" }, regions.Items.Where(i => i.IsSelected).Select(i => i.Name));
        Assert.Equal(1, changes); // one recomputation, not one per row
        Assert.Equal("7", regions.MapSpec!.Points[6].Label); // the map numbers its corners
    });

    [Fact]
    public Task Sectors_of_a_saved_track_are_listed_under_the_corners_as_soon_as_the_session_is_loaded() => Ui.Run(async () =>
    {
        var dir = Directory.CreateTempSubdirectory("inferno-regions");
        try
        {
            // A track with two gates, saved by an earlier run of the app.
            var path = Path.Combine(dir.FullName, "tracks.yaml");
            var first = new MainWindowViewModel(BuiltinOnly, new TrackStore(path));
            first.ActiveTab = first.Sectors;
            first.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
            await first.Sectors.RunNowAsync();
            first.Sectors.NewTrack();
            var n = first.Sectors.ReferencePath!.Value.X.Length;
            first.Sectors.AddGateAt(n / 3);
            first.Sectors.AddGateAt(2 * n / 3);
            first.Sectors.SaveTrack();

            var viewModel = new MainWindowViewModel(BuiltinOnly, new TrackStore(path));
            foreach (var tab in viewModel.Tabs)
            {
                tab.Debounce = TimeSpan.Zero;
            }

            viewModel.ActiveTab = viewModel.Suspension;
            var window = new MainWindow(viewModel) { Width = 1300, Height = 1100 };
            window.Show();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            viewModel.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
            await viewModel.Suspension.RunNowAsync();
            await viewModel.Regions.LastRefresh;
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "sidebar_regions_sectors.png");

            var panel = window.FindControl<RegionPanelView>("RegionPanel")!;
            var rows = panel.GetVisualDescendants().OfType<CheckBox>().Where(b => b.Classes.Contains("regionrow")).ToList();
            Assert.Equal(12, rows.Count); // nine corners and three sectors
            var scroll = panel.FindControl<ScrollViewer>("Rows")!;
            Assert.True(scroll.Extent.Height <= scroll.Viewport.Height + 0.5, $"the rows need {scroll.Extent.Height} and have {scroll.Viewport.Height}"); // a sidebar this tall shows every row
            Assert.All(rows, r => Assert.True(r.Bounds.Height > 0));
            var sessionA = window.FindControl<SessionPanelView>("SessionAPanel")!;
            var map = panel.FindControl<ChartHost>("Map")!;
            Assert.True(sessionA.Bounds.Height >= 300); // and the laps keep the rest
            Assert.True(panel.Bounds.Bottom <= window.FindControl<Grid>("Sidebar")!.Bounds.Height + 0.5); // the map is on screen, not below the window

            // A shorter window: the card gives way, its rows scroll, and the session keeps its laps.
            window.Height = 760;
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            window.CaptureRenderedFrame();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "sidebar_regions_sectors_short.png");
            Assert.True(scroll.Extent.Height > scroll.Viewport.Height);
            Assert.True(scroll.Viewport.Height >= 3 * 24 - 1, $"{scroll.Viewport.Height} of rows in sight");
            Assert.True(sessionA.Bounds.Height >= 280, $"session panel {sessionA.Bounds.Height}"); // its header and five or six laps
            Assert.True(map.IsEffectivelyVisible && map.Bounds.Height > 100);

            // Tall again: every row back in sight, the card no taller than it needs.
            window.Height = 1100;
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            window.CaptureRenderedFrame();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            window.CaptureRenderedFrame();
            Assert.True(scroll.Extent.Height <= scroll.Viewport.Height + 0.5);
            Assert.InRange(scroll.Viewport.Height - scroll.Extent.Height, -0.5, 1);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });

    [Fact]
    public Task The_sidebar_card_renders_and_a_copied_chart_names_its_regions() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var panel = window.FindControl<RegionPanelView>("RegionPanel")!;
        Assert.True(panel.IsVisible);
        var rows = panel.GetVisualDescendants().OfType<CheckBox>().Where(b => b.Classes.Contains("regionrow")).ToList();
        Assert.Equal(9, rows.Count); // one row per corner; no sectors without a saved track
        Assert.Matches(@"^\d+ km/h$", viewModel.Regions.Corners[2].Speed);
        Assert.Contains(viewModel.Regions.Corners[2].Direction, new[] { "L", "R" });
        Assert.Contains("lit", panel.FindControl<Button>("WholeLap")!.Classes); // nothing ticked: whole laps is the state

        var locator = (TabViewLocator)window.FindControl<TabControl>("TabArea")!.ContentTemplate!;
        var chart = locator.ViewOf(viewModel.Suspension)!.FindControl<ChartHost>("Chart")!;
        using var whole = chart.RenderImage()!;

        rows[2].IsChecked = true; // a click on Turn 3's row
        rows[4].IsChecked = true;
        await viewModel.Suspension.RunNowAsync();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal("Turn 3 + Turn 5", viewModel.Regions.Summary);
        Assert.DoesNotContain("lit", panel.FindControl<Button>("WholeLap")!.Classes);
        var tag = locator.ViewOf(viewModel.Suspension)!.GetVisualDescendants().OfType<RegionTagView>().Single();
        Assert.True(tag.IsVisible);
        RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "sidebar_regions.png");

        // The toolbar's tag is not part of a copied chart, so the image carries the regions itself.
        using var regional = chart.RenderImage()!;
        Assert.Equal(whole.PixelSize.Width, regional.PixelSize.Width);
        Assert.True(regional.PixelSize.Height > whole.PixelSize.Height);
        RenderSmokeTests.SaveRender(regional, "chart_image_with_regions.png");

        // On a tab that does not use regions the rows are dimmed, not gone: the sidebar does not jump.
        viewModel.ActiveTab = viewModel.Sectors;
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.True(panel.IsVisible);
        Assert.All(rows, c => Assert.False(c.IsEffectivelyEnabled));
        Assert.True(panel.FindControl<TextBlock>("UseCaption")!.IsVisible);

        // Two sessions: the card keeps its place and its map; the lap lists share what is left.
        viewModel.ActiveTab = viewModel.Suspension;
        viewModel.SessionB.Apply(Sample.Value, SessionParityTests.SamplePath());
        await viewModel.Suspension.RunNowAsync();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.True(panel.FindControl<ChartHost>("Map")!.IsEffectivelyVisible);
        RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "sidebar_regions_two_sessions.png");
    });

    [Fact]
    public Task The_map_shows_where_the_pointer_on_the_charts_is_on_the_track() => Ui.Run(async () =>
    {
        var viewModel = await LoadedWindow();
        var window = new MainWindow(viewModel) { Width = 1300, Height = 900 };
        window.Show();
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        var regions = viewModel.Regions;
        var map = (InfernoAnalyzer.Charting.ScottPlot.ScottPlotTrackMap)window.FindControl<RegionPanelView>("RegionPanel")!.FindControl<ChartHost>("Map")!.Surface!;
        var line = regions.MapSpec!.Cursor!;
        (double X, double Y) OnLap(double fraction)
        {
            var i = Array.FindIndex(line.Along, a => a >= fraction);
            return (line.X[i], line.Y[i]);
        }

        static double Metres((double X, double Y) a, (double X, double Y) b) => Math.Sqrt(((a.X - b.X) * (a.X - b.X)) + ((a.Y - b.Y) * (a.Y - b.Y)));
        Assert.Null(map.CursorPoint); // nothing under a pointer yet

        // The Time/Distance cursor, against distance: a quarter of the way round the reference lap.
        var traces = viewModel.TimeDistance;
        viewModel.ActiveTab = traces;
        await traces.RunNowAsync();
        var lapLength = traces.Spec!.Subplots[0].Traces[0].Emphasis!.Value.Max;
        traces.Cursors.SetCursor(0.25 * lapLength);
        Assert.Equal(0.25, regions.Position.Cursor!.Value, 6);
        Assert.InRange(Metres(map.CursorPoint!.Value, OnLap(0.25)), 0, 5);
        RenderSmokeTests.SaveRender(window.CaptureRenderedFrame()!, "sidebar_map_cursor.png");

        // On the lap stitched on after it, the same place one lap later; off the chart, no marker.
        traces.Cursors.SetCursor(1.25 * lapLength);
        Assert.Equal(0.25, regions.Position.Cursor!.Value, 6);
        traces.Cursors.SetCursor(null);
        Assert.Null(map.CursorPoint);

        // Against time, the reference lap's seconds are a place and the neighbours' are not.
        traces.ToggleXAxis();
        var lapTime = traces.Spec!.Subplots[0].Traces[0].Emphasis!.Value.Max;
        traces.Cursors.SetCursor(0.5 * lapTime);
        Assert.InRange(regions.Position.Cursor!.Value, 0.3, 0.7);
        Assert.NotNull(map.CursorPoint);
        traces.Cursors.SetCursor(1.5 * lapTime);
        Assert.Null(map.CursorPoint);
        traces.ToggleXAxis();

        // Another tab starts without the last one's marker. On Driver Consistency a corner's column
        // under the pointer is its apex, which is where the map has that corner's dot.
        traces.Cursors.SetCursor(0.25 * lapLength);
        var driver = viewModel.Driver;
        viewModel.ActiveTab = driver;
        await driver.RunNowAsync();
        Assert.Null(map.CursorPoint);
        driver.HoveredCorner = 2;
        var turn3 = regions.MapSpec!.Points[2];
        Assert.InRange(Metres(map.CursorPoint!.Value, (turn3.X, turn3.Y)), 0, 30);
        driver.HoveredCorner = -1;
        Assert.Null(map.CursorPoint);

        // The cursor of its detail view is metres along the lap.
        driver.OpenCorner(0);
        var turn1 = regions.MapSpec.Points[0];
        var apex = driver.DetailSpec!.Markers!.Single(m => m.Label == "Apex").X;
        driver.DetailCursor.SetCursor(apex);
        Assert.InRange(Metres(map.CursorPoint!.Value, (turn1.X, turn1.Y)), 0, 30);

        // A tab that is not showing does not move the marker.
        traces.Cursors.SetCursor(0.9 * lapLength);
        Assert.InRange(Metres(map.CursorPoint!.Value, (turn1.X, turn1.Y)), 0, 30);
    });
}
