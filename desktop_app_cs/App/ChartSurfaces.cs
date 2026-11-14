using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Platform;
using Avalonia.Styling;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Charting.ScottPlot;

namespace InfernoAnalyzer.App;

/// <summary>
/// The one place the chart backend is chosen. Swapping libraries means a new
/// adapter project and new lines here; views and viewmodels never change.
/// Also where the chart theme is derived from the app's own theme
/// resources, so charts sit on the same surfaces and use the same font as
/// the controls around them.
/// </summary>
public static class ChartSurfaces
{
    private static bool _fontsRegistered;

    /// <summary>Create the surface that renders <paramref name="spec"/>'s kind, themed for the app's current variant.</summary>
    public static IChartSurface CreateFor(ChartSpec spec) => spec switch
    {
        StripChartSpec => CreateStripChart(),
        HistogramGridSpec => CreateHistogramGrid(),
        ScatterGridSpec => CreateScatterGrid(),
        BoxPlotSpec => CreateBoxPlot(),
        TrackMapSpec => CreateTrackMap(),
        _ => throw new NotSupportedException($"No chart surface renders {spec.GetType().Name}"),
    };

    /// <summary>Create a surface able to render <see cref="StripChartSpec"/>.</summary>
    public static IChartSurface CreateStripChart() => Themed(new ScottPlotStripChart());

    /// <summary>Create a surface able to render <see cref="HistogramGridSpec"/>.</summary>
    public static IChartSurface CreateHistogramGrid() => Themed(new ScottPlotHistogramGrid());

    /// <summary>Create a surface able to render <see cref="ScatterGridSpec"/>.</summary>
    public static IChartSurface CreateScatterGrid() => Themed(new ScottPlotScatterGrid());

    /// <summary>Create a surface able to render <see cref="BoxPlotSpec"/>.</summary>
    public static IChartSurface CreateBoxPlot() => Themed(new ScottPlotBoxPlot());

    /// <summary>Create a surface able to render <see cref="TrackMapSpec"/>.</summary>
    public static IChartSurface CreateTrackMap() => Themed(new ScottPlotTrackMap());

    /// <summary>The chart theme matching the app's current variant.</summary>
    public static ChartTheme CurrentTheme => ThemeFor(Application.Current?.ActualThemeVariant);

    /// <summary>The chart theme for the variant <paramref name="host"/> is shown in.</summary>
    public static ChartTheme ThemeFor(StyledElement host) => ThemeFor(host.ActualThemeVariant);

    /// <summary>
    /// The chart theme for <paramref name="variant"/> (light for the light
    /// variant, dark otherwise): the built-in theme with its surfaces, ink
    /// and font taken from the app's resources where they are defined, or
    /// the high-contrast theme when the OS asks for more contrast.
    /// </summary>
    public static ChartTheme ThemeFor(ThemeVariant? variant)
    {
        var light = variant == ThemeVariant.Light;
        var app = Application.Current;
        var family = AppFonts.ChartFamily;
        if (app?.PlatformSettings?.GetColorValues().ContrastPreference == ColorContrastPreference.High)
        {
            return (light ? ChartTheme.HighContrastLight : ChartTheme.HighContrastDark) with { FontFamily = family };
        }

        var theme = light ? ChartTheme.Light : ChartTheme.Dark;
        if (app is null)
        {
            return theme with { FontFamily = family };
        }

        var key = light ? ThemeVariant.Light : ThemeVariant.Dark;
        string Token(string resource, string fallback) =>
            app.TryGetResource(resource, key, out var value) && value is ISolidColorBrush brush ? Hex(brush.Color) : fallback;
        return theme with
        {
            FontFamily = family,
            FigureBackground = Token("AppCardBrush", theme.FigureBackground),
            DataBackground = Token("AppChartDataBrush", theme.DataBackground),
            Grid = Token("AppChartGridBrush", theme.Grid),
            Axis = Token("AppSubtleTextBrush", theme.Axis),
            Text = Token("AppTextBrush", theme.Text),
            PanelBackground = Token("AppCardBrush", theme.PanelBackground),
        };
    }

    private static string Hex(Color color) => $"#{color.R:x2}{color.G:x2}{color.B:x2}";

    private static IChartSurface Themed(IChartSurface surface)
    {
        RegisterFonts();
        surface.Theme = CurrentTheme;
        return surface;
    }

    /// <summary>Hand the bundled UI font to the chart backend (once), so chart text matches the controls.</summary>
    private static void RegisterFonts()
    {
        if (_fontsRegistered)
        {
            return;
        }

        _fontsRegistered = true;
        if (AppFonts.UsesBundledInter)
        {
            ScottPlotFonts.Register(
                AppFonts.Inter,
                () => AssetLoader.Open(new Uri("avares://Avalonia.Fonts.Inter/Assets/Inter-Regular.ttf")),
                () => AssetLoader.Open(new Uri("avares://Avalonia.Fonts.Inter/Assets/Inter-SemiBold.ttf")));
        }
    }
}

/// <summary>
/// Which font the app draws in: the system UI font where the platform has a
/// good one (Segoe UI on Windows, San Francisco on macOS), the bundled Inter
/// on Linux, where "the system font" is whatever the distribution happens to
/// ship.
/// </summary>
public static class AppFonts
{
    /// <summary>Family name of the bundled font.</summary>
    public const string Inter = "Inter";

    /// <summary>Whether the app (and so the charts) should use the bundled Inter.</summary>
    public static bool UsesBundledInter => !OperatingSystem.IsWindows() && !OperatingSystem.IsMacOS();

    /// <summary>The family chart text is drawn in; empty leaves the chart library's default (the system sans).</summary>
    public static string ChartFamily =>
        OperatingSystem.IsWindows() ? "Segoe UI" : UsesBundledInter ? Inter : string.Empty;
}
