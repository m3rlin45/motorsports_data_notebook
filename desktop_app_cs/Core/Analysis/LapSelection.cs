using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Lap selection helpers; port of <c>channels.py</c>.</summary>
public static class LapSelection
{
    /// <summary>
    /// <c>get_top_laps</c>: laps within <paramref name="thresholdPct"/> of the
    /// best lap time. When the laps carry types only "full" laps qualify;
    /// otherwise the first and last laps are excluded (AIM files).
    /// </summary>
    public static List<Lap> GetTopLaps(IReadOnlyList<Lap> laps, double thresholdPct = 1.03)
    {
        var positive = laps.Where(l => l.LapTimeMs > 0).ToList();
        var valid = laps.Any(l => l.LapType.Length > 0)
            ? positive.Where(l => l.LapType == "full").ToList()
            : positive.Skip(1).Take(Math.Max(positive.Count - 2, 0)).ToList();
        if (valid.Count == 0)
        {
            return valid;
        }

        var best = valid.Min(l => l.LapTimeMs);
        var threshold = best * thresholdPct;
        return valid.Where(l => l.LapTimeMs <= threshold).ToList();
    }

    /// <summary>
    /// <c>get_best_lap</c>: the fastest full lap (or, without lap types, the
    /// fastest excluding the first and last).
    /// </summary>
    /// <exception cref="InvalidOperationException">No eligible lap.</exception>
    public static Lap GetBestLap(IReadOnlyList<Lap> laps)
    {
        List<Lap> subset;
        if (laps.Any(l => l.LapType.Length > 0))
        {
            subset = laps.Where(l => l.LapType == "full").ToList();
            if (subset.Count == 0)
            {
                throw new InvalidOperationException("No full laps found");
            }
        }
        else
        {
            if (laps.Count <= 2)
            {
                throw new InvalidOperationException("Need at least 3 laps to exclude first and last laps");
            }

            subset = laps.Skip(1).Take(laps.Count - 2).ToList();
        }

        return subset.MinBy(l => l.LapTimeMs)!;
    }
}
