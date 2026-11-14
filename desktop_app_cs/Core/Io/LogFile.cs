namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// A telemetry session: channels at their native rates, the laps table and
/// the file metadata. Immutable; the filter/select/resample operations return
/// new instances and follow libxrk's Python <c>LogFile</c> exactly, so the
/// analysis kernels see the same data as the notebooks.
/// </summary>
public sealed class LogFile
{
    private readonly Dictionary<string, Channel> _byName;

    /// <summary>Create a session from parsed parts; channel order is preserved.</summary>
    public LogFile(
        IReadOnlyList<Channel> channels,
        IReadOnlyList<Lap> laps,
        IReadOnlyDictionary<string, string> metadata,
        string fileName,
        string metadataJson = "{}")
    {
        Channels = channels;
        Laps = laps;
        Metadata = metadata;
        FileName = fileName;
        MetadataJson = metadataJson;
        _byName = new Dictionary<string, Channel>(StringComparer.Ordinal);
        foreach (var channel in channels)
        {
            _byName[channel.Name] = channel;
        }
    }

    /// <summary>Channels in parser order.</summary>
    public IReadOnlyList<Channel> Channels { get; }

    /// <summary>Laps in table order.</summary>
    public IReadOnlyList<Lap> Laps { get; }

    /// <summary>
    /// File metadata with every value as text: scalars as their string form
    /// (as Python's <c>str()</c> would give), nested values as JSON. The keys
    /// the profile lookup reads are <c>Logger ID</c> and <c>Vehicle</c>.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>The full metadata document as JSON.</summary>
    public string MetadataJson { get; }

    /// <summary>Path the session was loaded from, or a placeholder for bytes.</summary>
    public string FileName { get; }

    /// <summary>Channel names in order.</summary>
    public IEnumerable<string> ChannelNames => Channels.Select(c => c.Name);

    /// <summary>Whether a channel exists.</summary>
    public bool HasChannel(string name) => _byName.ContainsKey(name);

    /// <summary>Look up a channel.</summary>
    /// <exception cref="KeyNotFoundException">No such channel.</exception>
    public Channel this[string name] =>
        _byName.TryGetValue(name, out var channel)
            ? channel
            : throw new KeyNotFoundException($"Channel not found: {name}");

    /// <summary>Try to look up a channel.</summary>
    public bool TryGetChannel(string name, out Channel channel) => _byName.TryGetValue(name, out channel!);

    /// <summary>Only the named channels, in the given order.</summary>
    /// <exception cref="KeyNotFoundException">A name is missing.</exception>
    public LogFile SelectChannels(IEnumerable<string> names)
    {
        var wanted = names.ToList();
        var missing = wanted.Where(n => !_byName.ContainsKey(n)).OrderBy(n => n, StringComparer.Ordinal).ToList();
        if (missing.Count > 0)
        {
            throw new KeyNotFoundException($"Channels not found: [{string.Join(", ", missing)}]");
        }

        return With(wanted.Select(n => _byName[n]).ToList(), Laps);
    }

    /// <summary>
    /// Samples with <c>start &lt;= t &lt; end</c> in every channel, and the laps
    /// overlapping the range (<c>lap.Start &lt; end &amp;&amp; lap.End &gt; start</c>).
    /// </summary>
    public LogFile FilterByTimeRange(long start, long end, IEnumerable<string>? names = null)
    {
        var source = names is null ? this : SelectChannels(names);
        var channels = source.Channels.Select(c => c.Slice(start, end)).ToList();
        var laps = Laps.Where(l => l.StartTime < end && l.EndTime > start).ToList();
        return With(channels, laps);
    }

    /// <summary>The time range of lap <paramref name="num"/> (first matching row).</summary>
    /// <exception cref="ArgumentException">No such lap.</exception>
    public LogFile FilterByLap(int num, IEnumerable<string>? names = null)
    {
        var lap = Laps.FirstOrDefault(l => l.Num == num)
            ?? throw new ArgumentException($"Lap {num} not found. Available laps: [{string.Join(", ", Laps.Select(l => l.Num))}]");
        return FilterByTimeRange(lap.StartTime, lap.EndTime, names);
    }

    /// <summary>Every channel resampled onto <paramref name="timecodes"/> (see <see cref="Channel.ResampleTo"/>).</summary>
    public LogFile ResampleToTimecodes(long[] timecodes, IEnumerable<string>? names = null)
    {
        var source = names is null ? this : SelectChannels(names);
        return With(source.Channels.Select(c => c.ResampleTo(timecodes)).ToList(), Laps);
    }

    /// <summary>Every channel resampled onto the timecodes of <paramref name="reference"/>.</summary>
    /// <exception cref="KeyNotFoundException">No such reference channel.</exception>
    public LogFile ResampleToChannel(string reference, IEnumerable<string>? names = null)
    {
        if (!_byName.TryGetValue(reference, out var channel))
        {
            throw new KeyNotFoundException($"Reference channel not found: {reference}");
        }

        return ResampleToTimecodes(channel.Timecodes, names);
    }

    /// <summary>A copy with an extra (or replaced) channel appended.</summary>
    public LogFile WithChannel(Channel channel)
    {
        var channels = Channels.Where(c => c.Name != channel.Name).Append(channel).ToList();
        return With(channels, Laps);
    }

    /// <summary>A copy with a different laps table.</summary>
    public LogFile WithLaps(IReadOnlyList<Lap> laps) => With(Channels, laps);

    private LogFile With(IReadOnlyList<Channel> channels, IReadOnlyList<Lap> laps) =>
        new(channels, laps, Metadata, FileName, MetadataJson);
}
