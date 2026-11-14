namespace InfernoAnalyzer.Charting;

/// <summary>
/// How charts look under one UI theme: surfaces, ink, the type ramp and the
/// values of the <see cref="ChartColors"/> tokens. The app derives one from
/// its own theme resources so charts sit on the same surfaces as the controls
/// around them; every backend applies it the same way.
/// </summary>
/// <param name="FigureBackground">Outside the data area (the card the chart sits on).</param>
/// <param name="DataBackground">The data area.</param>
/// <param name="Axis">Axis lines, ticks, tick labels and axis captions.</param>
/// <param name="Grid">Grid lines and the axis frame.</param>
/// <param name="Text">Titles and label text.</param>
/// <param name="PanelBackground">Legend and label boxes.</param>
/// <param name="PanelOutline">Legend and label box outlines.</param>
/// <param name="Crosshair">Cursor line and markers.</param>
public sealed record ChartTheme(
    string FigureBackground,
    string DataBackground,
    string Axis,
    string Grid,
    string Text,
    string PanelBackground,
    string PanelOutline,
    string Crosshair)
{
    /// <summary>Whether the surfaces are dark.</summary>
    public bool IsDark { get; init; }

    /// <summary>
    /// Font family for chart text, matching the UI font; empty leaves the
    /// backend's default.
    /// </summary>
    public string FontFamily { get; init; } = string.Empty;

    /// <summary>Panel title size.</summary>
    public float TitleSize { get; init; } = 12;

    /// <summary>Axis caption, annotation and hover label size.</summary>
    public float LabelSize { get; init; } = 12;

    /// <summary>Tick label size.</summary>
    public float TickSize { get; init; } = 11;

    /// <summary>Multiplier on every data line width (raised in high-contrast mode).</summary>
    public float LineScale { get; init; } = 1;

    /// <summary>Whether this is a high-contrast theme: no translucency, heavier lines.</summary>
    public bool HighContrast { get; init; }

    /// <summary>Token → hex for the <see cref="ChartColors"/> tokens.</summary>
    public IReadOnlyDictionary<string, string> Palette { get; init; } = ChartColors.LightPalette;

    /// <summary>
    /// The hex value of <paramref name="color"/>: a <see cref="ChartColors"/>
    /// token is looked up in <see cref="Palette"/> (unknown tokens fall back
    /// to the text colour), anything else is returned unchanged.
    /// </summary>
    public string Resolve(string color)
    {
        if (color.Length == 0 || color[0] != '@')
        {
            return color;
        }

        if (color == ChartColors.Ink)
        {
            return Text;
        }

        return Palette.TryGetValue(color, out var hex) ? hex : Text;
    }

    /// <summary>Dark UI.</summary>
    public static ChartTheme Dark { get; } = new("#2b2b2b", "#1e1e1e", "#a6a6a6", "#333333", "#e8e8e8", "#2b2b2b", "#454545", "#f5f5f5")
    {
        IsDark = true,
        Palette = ChartColors.DarkPalette,
    };

    /// <summary>Light UI.</summary>
    public static ChartTheme Light { get; } = new("#fbfbfb", "#ffffff", "#5c5c5c", "#e5e5e5", "#1a1a1a", "#ffffff", "#d0d0d0", "#202020")
    {
        Palette = ChartColors.LightPalette,
    };

    /// <summary>High-contrast dark UI: white ink on black, heavier lines.</summary>
    public static ChartTheme HighContrastDark { get; } = new("#000000", "#000000", "#ffffff", "#5a5a5a", "#ffffff", "#000000", "#ffffff", "#ffffff")
    {
        IsDark = true,
        HighContrast = true,
        LineScale = 1.5f,
        Palette = Brighten(ChartColors.DarkPalette, "#ffffff"),
    };

    /// <summary>High-contrast light UI: black ink on white, heavier lines.</summary>
    public static ChartTheme HighContrastLight { get; } = new("#ffffff", "#ffffff", "#000000", "#a0a0a0", "#000000", "#ffffff", "#000000", "#000000")
    {
        HighContrast = true,
        LineScale = 1.5f,
        Palette = Brighten(ChartColors.LightPalette, "#000000"),
    };

    private static Dictionary<string, string> Brighten(IReadOnlyDictionary<string, string> palette, string ink) =>
        new(palette, StringComparer.Ordinal) { [ChartColors.Ink] = ink, [ChartColors.Neutral] = ink };
}
