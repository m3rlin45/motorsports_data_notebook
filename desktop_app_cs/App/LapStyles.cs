using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App;

/// <summary>
/// How one lap is drawn wherever it appears as a trace: one of the
/// <see cref="ChartColors.LapSlots"/> lap hues, solid for the first round of
/// slots and dashed for the second, so sixteen laps stay distinguishable.
/// </summary>
/// <param name="Slot">Style slot, 0 … <see cref="LapStyleAllocator.SlotCount"/> − 1.</param>
public readonly record struct LapStyle(int Slot)
{
    /// <summary>The lap's colour token.</summary>
    public string Color => ChartColors.Lap(Slot);

    /// <summary>Whether the lap's line is dashed (the second round through the hues).</summary>
    public bool Dashed => Slot / ChartColors.LapSlots % 2 == 1;
}

/// <summary>
/// Gives every selected lap of both sessions its own <see cref="LapStyle"/>
/// and keeps it for as long as the lap stays selected, so a lap has the same
/// colour on every tab and ticking another lap never repaints the laps
/// overlaid beside it. The lap list shows the style as the legend.
/// </summary>
/// <remarks>
/// Laps come in two tiers. The laps overlaid as traces are few and are
/// compared by eye, so they take the lowest slots (the solid hues) across
/// both sessions and never give theirs up. The laps the statistical tabs
/// aggregate only appear one-per-trace in the Driver tab's corner detail;
/// they fill the remaining slots and move aside when a new trace lap needs
/// one.
/// </remarks>
public sealed class LapStyleAllocator
{
    /// <summary>Distinct styles before they repeat: every hue solid, then every hue dashed.</summary>
    public const int SlotCount = ChartColors.LapSlots * 2;

    private readonly Dictionary<(string Session, int Lap), int> _slots = new();
    private readonly Dictionary<string, (List<int> Traces, List<int> Others)> _sessions = new();

    /// <summary>Raised after any assignment: a lap of either session may have changed style.</summary>
    public event Action? Changed;

    /// <summary>
    /// Set the styled laps of <paramref name="session"/>: its
    /// <paramref name="traces"/> (the overlaid laps, in priority order) and
    /// the <paramref name="others"/> that are selected elsewhere. Laps that
    /// already have a style keep it where the tiers allow; laps no longer
    /// listed release theirs.
    /// </summary>
    public void Assign(string session, IReadOnlyList<int> traces, IReadOnlyList<int> others)
    {
        _sessions[session] = (traces.ToList(), others.Except(traces).ToList());
        var sessions = _sessions.OrderBy(s => s.Key, StringComparer.Ordinal).ToList();
        var traceKeys = sessions.SelectMany(s => s.Value.Traces.Select(lap => (s.Key, lap))).ToList();
        var otherKeys = sessions.SelectMany(s => s.Value.Others.Select(lap => (s.Key, lap))).ToList();

        foreach (var stale in _slots.Keys.Except(traceKeys).Except(otherKeys).ToList())
        {
            _slots.Remove(stale);
        }

        // Trace laps: keep what they have; a new one takes the lowest slot no other trace lap holds,
        // evicting an aggregated lap from it if need be.
        var taken = traceKeys.Where(_slots.ContainsKey).Select(k => _slots[k]).ToHashSet();
        foreach (var key in traceKeys.Where(k => !_slots.ContainsKey(k)))
        {
            var slot = LowestFree(taken) ?? taken.Count % SlotCount; // more trace laps than styles: reuse
            foreach (var evicted in otherKeys.Where(k => _slots.TryGetValue(k, out var s) && s == slot).ToList())
            {
                _slots.Remove(evicted);
            }

            _slots[key] = slot;
            taken.Add(slot);
        }

        // Aggregated laps fill what is left, in the order given.
        var used = _slots.Values.ToHashSet();
        foreach (var key in otherKeys.Where(k => !_slots.ContainsKey(k)))
        {
            var slot = LowestFree(used) ?? _slots.Count % SlotCount;
            _slots[key] = slot;
            used.Add(slot);
        }

        Changed?.Invoke();
    }

    private static int? LowestFree(HashSet<int> used)
    {
        for (var slot = 0; slot < SlotCount; slot++)
        {
            if (!used.Contains(slot))
            {
                return slot;
            }
        }

        return null;
    }

    /// <summary>The style of a lap, or null when it is not selected anywhere.</summary>
    public LapStyle? Get(string session, int lap) =>
        _slots.TryGetValue((session, lap), out var slot) ? new LapStyle(slot) : null;
}
