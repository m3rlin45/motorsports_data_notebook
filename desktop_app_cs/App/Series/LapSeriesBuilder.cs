using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.App.Series;

/// <summary>
/// Builds the per-lap series the strip chart draws from a loaded session:
/// the chosen channels resampled onto one timebase for each lap, with the
/// neighbouring laps attached so the view can scroll past the lap. The
/// result reuses the <see cref="Fixture"/> shape that
/// <c>tools/extract_fixture.py</c> dumps, so the chart code has one input.
/// </summary>
public static class LapSeriesBuilder
{
    /// <summary>
    /// Extract <paramref name="channels"/> for <paramref name="lapNums"/> (draw
    /// order; the first is the delta-time reference) on the timecodes of
    /// <paramref name="timebase"/>. Channels missing from the session are skipped.
    /// <paramref name="cancellation"/> is checked between laps.
    /// </summary>
    public static Fixture Build(LogFile log, IEnumerable<int> lapNums, IReadOnlyList<string> channels, string timebase, CancellationToken cancellation = default)
    {
        var present = channels.Where(log.HasChannel).ToList();
        var fixture = new Fixture
        {
            Source = Path.GetFileName(log.FileName),
            Timebase = timebase,
            Channels = present,
        };
        foreach (var name in present)
        {
            fixture.Units[name] = log[name].Units;
        }

        var cache = new Dictionary<int, FixtureLap>();
        foreach (var num in lapNums)
        {
            cancellation.ThrowIfCancellationRequested();
            var lap = ExtractLap(log, num, present, timebase, cache);
            if (lap is null)
            {
                continue;
            }

            lap.Before = ExtractLap(log, num - 1, present, timebase, cache);
            lap.After = ExtractLap(log, num + 1, present, timebase, cache);
            fixture.Laps.Add(lap);
        }

        return fixture;
    }

    private static FixtureLap? ExtractLap(LogFile log, int num, IReadOnlyList<string> channels, string timebase, Dictionary<int, FixtureLap> cache)
    {
        if (cache.TryGetValue(num, out var cached))
        {
            return cached;
        }

        var lapRow = log.Laps.FirstOrDefault(l => l.Num == num);
        if (lapRow is null || lapRow.LapTimeMs <= 0)
        {
            return null;
        }

        var wanted = channels.Append("distance_m").Append(timebase).Distinct().ToList();
        if (!wanted.All(log.HasChannel))
        {
            return null;
        }

        var aligned = log.FilterByLap(num).SelectChannels(wanted).ResampleToChannel(timebase);
        var timecodes = aligned[timebase].Timecodes;
        if (timecodes.Length == 0)
        {
            return null;
        }

        var t0 = timecodes[0];
        var lap = new FixtureLap
        {
            Num = num,
            LapTimeS = lapRow.LapTimeS,
            TimeS = timecodes.Select(t => (t - t0) / 1000.0).ToArray(),
            DistanceM = aligned["distance_m"].Values,
        };
        foreach (var name in channels)
        {
            lap.Series[name] = aligned[name].Values;
        }

        cache[num] = lap;
        return lap;
    }
}
