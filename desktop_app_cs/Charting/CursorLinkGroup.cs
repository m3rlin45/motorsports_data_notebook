namespace InfernoAnalyzer.Charting;

/// <summary>
/// Keeps the crosshair of every member surface in lockstep: when one surface
/// reports a cursor move, the same x is pushed to all the others. Lives above
/// <see cref="IChartSurface"/>, so backends only need "draw a line at x" and
/// "tell me where the pointer is"; the sync logic is written once.
///
/// The group also exposes the current cursor position so non-chart consumers
/// (a readout panel, a track-map dot) can follow it.
/// </summary>
public sealed class CursorLinkGroup : IDisposable
{
    private readonly List<IChartSurface> _members = new();
    private readonly Dictionary<IChartSurface, Action<double?>> _handlers = new();
    private bool _broadcasting;

    /// <summary>The last cursor x reported by any member, or null when hidden.</summary>
    public double? Cursor { get; private set; }

    /// <summary>Raised after every cursor change, including programmatic ones.</summary>
    public event Action<double?>? CursorChanged;

    /// <summary>Add a surface to the group and start following its cursor.</summary>
    public void Add(IChartSurface surface)
    {
        if (_handlers.ContainsKey(surface))
        {
            return;
        }

        void Handler(double? x) => Broadcast(x, origin: surface);
        _handlers[surface] = Handler;
        _members.Add(surface);
        surface.CursorMoved += Handler;
    }

    /// <summary>Remove a surface from the group.</summary>
    public void Remove(IChartSurface surface)
    {
        if (_handlers.Remove(surface, out var handler))
        {
            surface.CursorMoved -= handler;
            _members.Remove(surface);
        }
    }

    /// <summary>Set the cursor on every member (e.g. from a track-map hover).</summary>
    public void SetCursor(double? x) => Broadcast(x, origin: null);

    private void Broadcast(double? x, IChartSurface? origin)
    {
        if (_broadcasting)
        {
            return; // a member echoed SetCursor back as CursorMoved; ignore
        }

        _broadcasting = true;
        try
        {
            Cursor = x;
            foreach (var member in _members)
            {
                if (!ReferenceEquals(member, origin))
                {
                    member.SetCursor(x);
                }
            }
        }
        finally
        {
            _broadcasting = false;
        }

        CursorChanged?.Invoke(x);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var (surface, handler) in _handlers)
        {
            surface.CursorMoved -= handler;
        }

        _handlers.Clear();
        _members.Clear();
    }
}
