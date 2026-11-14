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
    /// </summary>
    public static Fixture Build(LogFile log, IEnumerable<int> lapNums, IReadOnlyList<string> channels, string timebase) =>
        Build(log, lapNums, channels, timebase, string.Empty, null);

    /// <summary>
    /// As <see cref="Build(LogFile, IEnumerable{int}, IReadOnlyList{string}, string)"/>, tagging
    /// every lap with <paramref name="session"/>. <paramref name="channelMap"/> translates the
    /// display channel names (and the timebase) to this session's own names, so a comparison
    /// session from another vehicle lines up with Session A's panels; unmapped names are
    /// used as they are. <paramref name="cancellation"/> is checked between laps.
    /// </summary>
    public static Fixture Build(LogFile log, IEnumerable<int> lapNums, IReadOnlyList<string> channels, string timebase, string session, IReadOnlyDictionary<string, string>? channelMap, CancellationToken cancellation = default)
    {
        string Local(string name) => channelMap is not null && channelMap.TryGetValue(name, out var mapped) ? mapped : name;
        var localTimebase = Local(timebase);
        var present = channels.Where(c => log.HasChannel(Local(c))).ToList();
        var fixture = new Fixture
        {
            Source = Path.GetFileName(log.FileName),
            Timebase = timebase,
            Channels = present,
        };
        foreach (var name in present)
        {
            fixture.Units[name] = log[Local(name)].Units;
        }

        var cache = new Dictionary<int, FixtureLap>();
        foreach (var num in lapNums)
        {
            cancellation.ThrowIfCancellationRequested();
            var lap = ExtractLap(log, num, present, Local, localTimebase, session, cache);
            if (lap is null)
            {
                continue;
            }

            lap.Before = ExtractLap(log, num - 1, present, Local, localTimebase, session, cache);
            lap.After = ExtractLap(log, num + 1, present, Local, localTimebase, session, cache);
            fixture.Laps.Add(lap);
        }

        return fixture;
    }

    /// <summary>
    /// Session A's laps and Session B's in one fixture, fastest lap first (the
    /// delta-time reference). Panels are the display channels of A; a channel B
    /// lacks is simply absent from B's traces.
    /// </summary>
    public static Fixture Merge(Fixture a, Fixture b)
    {
        var merged = new Fixture
        {
            Source = $"{a.Source} + {b.Source}",
            Timebase = a.Timebase,
            Channels = a.Channels.Union(b.Channels).ToList(),
        };
        foreach (var (name, units) in b.Units.Concat(a.Units))
        {
            merged.Units[name] = units;
        }

        merged.Laps.AddRange(a.Laps.Concat(b.Laps).OrderBy(lap => lap.LapTimeS));
        return merged;
    }

    private static FixtureLap? ExtractLap(LogFile log, int num, IReadOnlyList<string> channels, Func<string, string> local, string timebase, string session, Dictionary<int, FixtureLap> cache)
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

        var wanted = channels.Select(local).Append("distance_m").Append(timebase).Distinct().ToList();
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
            Session = session,
            LapTimeS = lapRow.LapTimeS,
            TimeS = timecodes.Select(t => (t - t0) / 1000.0).ToArray(),
            DistanceM = aligned["distance_m"].Values,
        };
        foreach (var name in channels)
        {
            lap.Series[name] = aligned[local(name)].Values;
        }

        cache[num] = lap;
        return lap;
    }
}
