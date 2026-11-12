using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
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
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static async Task<MainWindowViewModel> LoadedWindow()
    {
        var window = new MainWindowViewModel(BuiltinOnly);
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
    public async Task Roll_gradient_tab_builds_the_map_and_three_panels_from_the_profile()
    {
        var window = await LoadedWindow();
        var tab = window.RollGradient;

        Assert.Equal("Analysis complete", tab.Status);
        Assert.Equal("1520", tab.RollConfig.WidthsA.Front);
        Assert.Equal("1540", tab.RollConfig.WidthsA.Rear);
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
    }

    [Fact]
    public async Task Missing_track_widths_block_the_analysis_with_a_message()
    {
        var window = await LoadedWindow();
        var tab = window.RollGradient;

        tab.RollConfig.WidthsA.Front = string.Empty;
        await tab.RunNowAsync();

        Assert.Equal("Enter front and rear track widths (mm)", tab.Status);
    }

    [AvaloniaFact]
    public async Task Hovering_the_track_map_highlights_samples_and_the_view_renders()
    {
        var viewModel = await LoadedWindow();
        var view = new RollGradientView { DataContext = viewModel.RollGradient };
        var window = new Window { Width = 1280, Height = 860, Content = view };
        window.Show();
        window.CaptureRenderedFrame();

        var surface = (ScottPlotScatterGrid)view.Chart;
        var path = viewModel.RollGradient.Spec!.Paths[0];
        surface.HighlightNear(path.X[100], path.Y[100]);
        Assert.True(surface.HighlightedCount > 0);
        var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, "roll_gradient.png"), new PngBitmapEncoderOptions());

        surface.ClearHighlights();
        Assert.Equal(0, surface.HighlightedCount);

        // A real pointer move far from the track clears; near it highlights.
        window.MouseMove(new Point(5, 5));
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(0, surface.HighlightedCount);
    }
}
