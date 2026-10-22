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
}
