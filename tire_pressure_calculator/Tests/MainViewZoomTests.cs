using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using TirePressureCalculator.ViewModels;
using TirePressureCalculator.Views;

namespace TirePressureCalculator.Tests;

public class MainViewZoomTests
{
    // Bug 1 repro: on Android phone mode, tapping a field zooms into its
    // quadrant via RenderTransform scale. Text selection handles (the nubs
    // that bracket the selected text) are placed in screen coordinates that
    // don't follow the RenderTransform, so they float above and to the left
    // of the actual text.
    //
    // Correct behavior: the zoom must be applied via a layout-affecting
    // transform (LayoutTransformControl + LayoutTransform) so the entire
    // measure/arrange pipeline scales, and TextBox adorners naturally
    // follow. The testable signature: when zoomed, the content is arranged
    // at the scaled size, so ScrollViewer.Extent is larger than Viewport.
    [AvaloniaFact]
    public void ZoomToQuadrant_ContentIsArrangedAtScaledSize()
    {
        var view = new MainView { DataContext = new MainViewModel() };
        var window = new Window
        {
            Width = 400,
            Height = 800,
            Content = view,
        };
        window.Show();
        Dispatcher.UIThread.RunJobs();

        var fl = view.FindControl<ContentControl>("FLCorner");
        var scroll = view.FindControl<ScrollViewer>("RootScroll");
        Assert.NotNull(fl);
        Assert.NotNull(scroll);

        // Baseline: before zoom, content fits viewport.
        var baselineExtent = scroll.Extent;

        view.ZoomTo(fl);
        Dispatcher.UIThread.RunJobs();

        // With a layout-affecting zoom, the ScrollViewer reports a larger
        // extent than viewport (content was arranged at 2x size). With the
        // buggy RenderTransform-only approach, extent == viewport because
        // the scale is purely visual and doesn't participate in layout.
        Assert.True(
            scroll.Extent.Width > scroll.Viewport.Width * 1.5,
            $"Expected extent > 1.5x viewport after zoom. " +
            $"Got Extent={scroll.Extent}, Viewport={scroll.Viewport}, " +
            $"BaselineExtent={baselineExtent}. " +
            $"This indicates the zoom uses RenderTransform (bug 1) rather " +
            $"than a layout-affecting transform.");
    }
}
