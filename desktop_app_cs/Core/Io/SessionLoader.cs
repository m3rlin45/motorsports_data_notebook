using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// <c>load_session</c> from the Python package: parse the file, add the
/// derived channels (<c>speed_kmh</c>, per-lap <c>distance_m</c>) and keep
/// the requested lap types, deduplicated.
/// </summary>
public static class SessionLoader
{
    /// <summary>Lap types kept by default: full laps only.</summary>
    public static readonly IReadOnlyList<string> DefaultLapTypes = new[] { "full" };

    /// <summary>The first bytes of an iRacing IBT file (<c>_IBT_MAGIC</c> in Python).</summary>
    private static ReadOnlySpan<byte> IbtMagic => new byte[] { 0x02, 0x00, 0x00, 0x00 };

    /// <summary>Load and post-process <paramref name="path"/>: .ibt via libibt, anything else via libxrk.</summary>
    public static LogFile Load(string path, ProfileStore? profiles = null, IReadOnlyList<string>? lapTypes = null)
    {
        var raw = path.EndsWith(".ibt", StringComparison.OrdinalIgnoreCase) ? IbtReader.Read(path) : XrkReader.Read(path);
        return PostProcess(raw, profiles ?? new ProfileStore(), lapTypes ?? DefaultLapTypes);
    }

    /// <summary>Load and post-process file bytes, sniffing the IBT magic as Python does.</summary>
    public static LogFile Load(ReadOnlySpan<byte> data, string fileName = "<bytes>", ProfileStore? profiles = null, IReadOnlyList<string>? lapTypes = null)
    {
        var raw = data.Length >= 4 && data[..4].SequenceEqual(IbtMagic) ? IbtReader.Read(data, fileName) : XrkReader.Read(data, fileName);
        return PostProcess(raw, profiles ?? new ProfileStore(), lapTypes ?? DefaultLapTypes);
    }

    /// <summary>
    /// Apply the derived channels and lap cleaning to a parsed session. The
    /// speed channel is resolved from the profile by logger ID alone, as the
    /// Python code does (it does not consult the vehicle name here).
    /// </summary>
    public static LogFile PostProcess(LogFile log, ProfileStore profiles, IReadOnlyList<string> lapTypes)
    {
        var speedName = ChannelKeys.DefaultChannelNames["gps_speed"];
        var loggerId = LogIdentity.GetLoggerId(log.Metadata);
        if (loggerId is not null && profiles.GetProfileForLogger(loggerId) is { } profile
            && profile.ChannelNames.TryGetValue("gps_speed", out var fromProfile))
        {
            speedName = fromProfile;
        }

        var result = log;
        if (log.TryGetChannel(speedName, out var speed))
        {
            var kmh = new double[speed.Length];
            for (var i = 0; i < kmh.Length; i++)
            {
                kmh[i] = speed.Values[i] * 3.6;
            }

            result = result.WithChannel(new Channel("speed_kmh", speed.Timecodes, kmh));
            result = result.WithChannel(new Channel("distance_m", speed.Timecodes, DistancePerLap(speed, log.Laps)));
        }

        var laps = result.Laps;
        if (laps.Count > 0 && laps.Any(l => l.LapType.Length > 0))
        {
            laps = laps.Where(l => lapTypes.Contains(l.LapType)).ToList();
        }

        laps = CleanLaps(laps, dropLapZero: !lapTypes.Contains("out"));
        return result.WithLaps(laps);
    }

    /// <summary>
    /// Cumulative distance within each lap (zero elsewhere), integrating speed
    /// (m/s) over the sample intervals: <c>cumsum(speed * dt)</c> with the
    /// first interval zero. Laps are applied in table order, so an overlapping
    /// later row overwrites an earlier one, as in Python.
    /// </summary>
    public static double[] DistancePerLap(Channel speed, IReadOnlyList<Lap> laps)
    {
        var t = speed.Timecodes;
        var v = speed.Values;
        var distance = new double[t.Length];
        foreach (var lap in laps)
        {
            var from = Channel.LowerBound(t, lap.StartTime);
            var to = Channel.LowerBound(t, lap.EndTime);
            if (to <= from)
            {
                continue;
            }

            var sum = 0.0;
            distance[from] = 0.0;
            for (var i = from + 1; i < to; i++)
            {
                // Python: dt in seconds from a cumulative time-since-start series.
                var dt = ((t[i] - t[from]) / 1000.0) - ((t[i - 1] - t[from]) / 1000.0);
                sum += v[i] * dt;
                distance[i] = sum;
            }
        }

        return distance;
    }

    /// <summary>
    /// <c>clean_laps</c>: drop lap 0 (unless out-laps are wanted) and keep the
    /// longest row per lap number, in table order. Returns the input unchanged
    /// if nothing would be left.
    /// </summary>
    public static IReadOnlyList<Lap> CleanLaps(IReadOnlyList<Lap> laps, bool dropLapZero = true)
    {
        if (laps.Count == 0)
        {
            return laps;
        }

        var best = new Dictionary<int, (int Index, long Duration)>();
        for (var i = 0; i < laps.Count; i++)
        {
            var lap = laps[i];
            if (lap.Num == 0 && dropLapZero)
            {
                continue;
            }

            if (!best.TryGetValue(lap.Num, out var current) || lap.LapTimeMs > current.Duration)
            {
                best[lap.Num] = (i, lap.LapTimeMs);
            }
        }

        if (best.Count == 0)
        {
            return laps;
        }

        return best.Values.Select(b => b.Index).OrderBy(i => i).Select(i => laps[i]).ToList();
    }
}
