using Avalonia;
using Avalonia.Controls;
using Avalonia.Styling;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Charting.ScottPlot;

namespace InfernoAnalyzer.App;

/// <summary>
/// The one place the chart backend is chosen. Swapping libraries means a new
/// adapter project and a new line here; views and viewmodels never change.
/// Surfaces created here follow the application's actual theme variant.
/// </summary>
public static class ChartSurfaces
{
    /// <summary>Create a surface able to render <see cref="StripChartSpec"/>.</summary>
    public static IChartSurface CreateStripChart() => FollowTheme(new ScottPlotStripChart());

    /// <summary>Create a surface able to render <see cref="HistogramGridSpec"/>.</summary>
    public static IChartSurface CreateHistogramGrid() => FollowTheme(new ScottPlotHistogramGrid());

    /// <summary>Create a surface able to render <see cref="ScatterGridSpec"/>.</summary>
    public static IChartSurface CreateScatterGrid() => FollowTheme(new ScottPlotScatterGrid());

    /// <summary>Create a surface able to render <see cref="BoxPlotSpec"/>.</summary>
    public static IChartSurface CreateBoxPlot() => FollowTheme(new ScottPlotBoxPlot());

    /// <summary>Create a surface able to render <see cref="TrackMapSpec"/>.</summary>
    public static IChartSurface CreateTrackMap() => FollowTheme(new ScottPlotTrackMap());

    /// <summary>Create a surface able to render <see cref="LineGridSpec"/>.</summary>
    public static IChartSurface CreateLineGrid() => FollowTheme(new ScottPlotLineGrid());

    /// <summary>The chart theme matching the app's current (OS-driven) variant.</summary>
    public static ChartTheme CurrentTheme => ThemeFor(Application.Current?.ActualThemeVariant);

    /// <summary>Light for the light variant, dark otherwise.</summary>
    public static ChartTheme ThemeFor(ThemeVariant? variant) => variant == ThemeVariant.Light ? ChartTheme.Light : ChartTheme.Dark;

    private static IChartSurface FollowTheme(IChartSurface surface)
    {
        surface.Theme = CurrentTheme;
        if (Application.Current is { } app)
        {
            // The surface lives as long as its view; the app outlives both, so detach on unload.
            EventHandler handler = (_, _) => surface.Theme = CurrentTheme;
            app.ActualThemeVariantChanged += handler;
            surface.Control.DetachedFromLogicalTree += (_, _) => app.ActualThemeVariantChanged -= handler;
            surface.Control.AttachedToLogicalTree += (_, _) =>
            {
                app.ActualThemeVariantChanged -= handler;
                app.ActualThemeVariantChanged += handler;
                surface.Theme = CurrentTheme;
            };
        }

        return surface;
    }
}
