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

        // The statistical tabs cover only the ticked regions and the Driver tab lists the corners
        // inside them; the others show whole laps and the selector says so.
        foreach (var tab in window.Tabs)
        {
            window.ActiveTab = tab;
            var uses = tab is SuspensionViewModel or RollGradientViewModel or TireGripViewModel or DriverViewModel;
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
}
