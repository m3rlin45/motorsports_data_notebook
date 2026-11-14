namespace InfernoAnalyzer.Charting;

/// <summary>
/// Semantic chart colours. Specs carry these tokens (or a literal hex) and a
/// backend resolves them through <see cref="ChartTheme.Resolve"/>, so the same
/// spec draws with the light or the dark step of each hue and never needs
/// rebuilding when the theme changes.
/// </summary>
public static class ChartColors
{
    /// <summary>Number of distinct lap hues; laps past this repeat a hue with a dashed line.</summary>
    public const int LapSlots = 8;

    /// <summary>Session A (the primary session) in aggregated views.</summary>
    public const string SessionA = "@session-a";

    /// <summary>Session B (the comparison session) in aggregated views.</summary>
    public const string SessionB = "@session-b";

    /// <summary>Bump (positive shock velocity).</summary>
    public const string Bump = "@bump";

    /// <summary>Rebound (negative shock velocity).</summary>
    public const string Rebound = "@rebound";

    /// <summary>Braking zones.</summary>
    public const string Braking = "@braking";

    /// <summary>Acceleration zones.</summary>
    public const string Acceleration = "@acceleration";

    /// <summary>Largest-opportunity highlight (bands, stars).</summary>
    public const string Opportunity = "@opportunity";

    /// <summary>Brake-point marker of a corner detail.</summary>
    public const string MarkerBrake = "@marker-brake";

    /// <summary>Corner entry/exit markers of a corner detail.</summary>
    public const string MarkerCorner = "@marker-corner";

    /// <summary>Apex marker of a corner detail.</summary>
    public const string MarkerApex = "@marker-apex";

    /// <summary>Hover highlight of Session A's samples.</summary>
    public const string HighlightA = "@highlight-a";

    /// <summary>Hover highlight of Session B's samples.</summary>
    public const string HighlightB = "@highlight-b";

    /// <summary>Warning text.</summary>
    public const string Warning = "@warning";

    /// <summary>Recessive marks: a track outline, a start/finish line.</summary>
    public const string Neutral = "@neutral";

    /// <summary>The theme's primary text colour, for marks that must read as ink.</summary>
    public const string Ink = "@ink";

    /// <summary>Friction velocity band.</summary>
    public const string BandFriction = "@band-friction";

    /// <summary>Low-speed velocity band.</summary>
    public const string BandSlow = "@band-slow";

    /// <summary>High-speed velocity band.</summary>
    public const string BandFast = "@band-fast";

    /// <summary>Curb-strike velocity band.</summary>
    public const string BandCurb = "@band-curb";

    private static readonly string[] LapTokens = Enumerable.Range(0, LapSlots).Select(i => $"@lap-{i}").ToArray();

    /// <summary>The hue of lap slot <paramref name="slot"/> (wraps past <see cref="LapSlots"/>).</summary>
    public static string Lap(int slot) => LapTokens[((slot % LapSlots) + LapSlots) % LapSlots];

    // The categorical lap hues are the dataviz reference palette in its validated
    // order (blue, orange, aqua, yellow, magenta, green, violet, red), stepped per
    // surface so every slot clears 3:1 against the data area: light steps on
    // #ffffff, dark steps on #1e1e1e. Re-run the palette validator after any edit.
    private static readonly string[] LapLight = { "#2a78d6", "#e2581f", "#12946a", "#b88000", "#d55181", "#008300", "#6250d6", "#d93a3a" };
    private static readonly string[] LapDark = { "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767" };

    /// <summary>Token values on a light surface.</summary>
    public static IReadOnlyDictionary<string, string> LightPalette { get; } = Build(
        LapLight,
        ink: "#1a1a1a",
        neutral: "#8a8a8a",
        acceleration: "#008300",
        highlightA: "#6250d6",
        highlightB: "#d55181",
        warning: "#c42b1c",
        markerBrake: "#12946a",
        opportunity: "#b88000");

    /// <summary>Token values on a dark surface.</summary>
    public static IReadOnlyDictionary<string, string> DarkPalette { get; } = Build(
        LapDark,
        ink: "#e8e8e8",
        neutral: "#8a8a8a",
        acceleration: "#2fa84f",
        highlightA: "#ffe066",
        highlightB: "#ff6ec7",
        warning: "#ff8080",
        markerBrake: "#199e70",
        opportunity: "#e0a41a");

    private static Dictionary<string, string> Build(
        string[] laps,
        string ink,
        string neutral,
        string acceleration,
        string highlightA,
        string highlightB,
        string warning,
        string markerBrake,
        string opportunity)
    {
        var palette = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [SessionA] = laps[0],
            [SessionB] = laps[1],
            [Bump] = laps[0],
            [Rebound] = laps[7],
            [Braking] = laps[7],
            [Acceleration] = acceleration,
            [Opportunity] = opportunity,
            [MarkerBrake] = markerBrake,
            [MarkerCorner] = laps[3],
            [MarkerApex] = laps[7],
            [HighlightA] = highlightA,
            [HighlightB] = highlightB,
            [Warning] = warning,
            [Neutral] = neutral,
            [Ink] = ink,
            // Band fills are drawn at low alpha behind the bars, the same tints on both surfaces.
            [BandFriction] = "#808080",
            [BandSlow] = "#6fb1dc",
            [BandFast] = "#5fc46a",
            [BandCurb] = "#e36a6a",
        };
        for (var i = 0; i < LapSlots; i++)
        {
            palette[LapTokens[i]] = laps[i];
        }

        return palette;
    }
}
