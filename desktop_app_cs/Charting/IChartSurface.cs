using Avalonia.Controls;

namespace InfernoAnalyzer.Charting;

/// <summary>
/// A rendering backend for one chart. The app only ever talks to this
/// interface; the concrete type comes from a factory in the app project.
///
/// Everything crosses this boundary in data coordinates (metres, seconds,
/// channel units) — never pixels — so the linked-cursor logic in
/// <see cref="CursorLinkGroup"/> is backend-independent by construction.
/// </summary>
public interface IChartSurface : IDisposable
{
    /// <summary>The Avalonia control to place in the visual tree.</summary>
    Control Control { get; }

    /// <summary>
    /// Replace the chart contents with <paramref name="spec"/>. Idempotent:
    /// calling it twice with the same spec leaves the chart unchanged.
    /// Throws <see cref="NotSupportedException"/> for spec kinds the backend
    /// does not implement.
    /// </summary>
    void Render(ChartSpec spec);

    /// <summary>
    /// Move the crosshair to x (data units) or hide it when null. Must not
    /// raise <see cref="CursorMoved"/>.
    /// </summary>
    void SetCursor(double? x);

    /// <summary>
    /// Raised when the user moves the pointer over the chart (x in data
    /// units) or leaves it (null).
    /// </summary>
    event Action<double?>? CursorMoved;

    /// <summary>Raised when the user clicks the chart.</summary>
    event Action<ChartPoint>? Clicked;

    /// <summary>Colours for the current UI theme; setting it re-renders the current spec.</summary>
    ChartTheme Theme { get; set; }
}

/// <summary>Chart colours for one UI theme (the app picks Light or Dark to match the OS).</summary>
/// <param name="FigureBackground">Outside the data area.</param>
/// <param name="DataBackground">The data area.</param>
/// <param name="Axis">Axis lines, ticks and labels.</param>
/// <param name="Grid">Grid lines.</param>
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
    /// <summary>Dark UI.</summary>
    public static ChartTheme Dark { get; } = new("#1e1e1e", "#262626", "#c8c8c8", "#3c3c3c", "#e8e8e8", "#2a2a2a", "#505050", "#f5f5f5");

    /// <summary>Light UI.</summary>
    public static ChartTheme Light { get; } = new("#f3f3f3", "#ffffff", "#404040", "#dcdcdc", "#1a1a1a", "#ffffff", "#b0b0b0", "#202020");
}
