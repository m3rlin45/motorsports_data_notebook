namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// A vehicle's channel name mapping, motion ratios and geometry. Port of the
/// Python <c>VehicleProfile</c>.
/// </summary>
public sealed class VehicleProfile
{
    /// <summary>Create a profile; missing channel keys are filled from <see cref="ChannelKeys.DefaultChannelNames"/>.</summary>
    public VehicleProfile(
        string name,
        IReadOnlyDictionary<string, string>? channelNames = null,
        MotionRatios? motionRatios = null,
        ShockCalibration? shockCalibration = null,
        TrackWidths? trackWidths = null)
    {
        Name = name;
        var names = new Dictionary<string, string>(ChannelKeys.DefaultChannelNames, StringComparer.Ordinal);
        if (channelNames is not null)
        {
            foreach (var (key, value) in channelNames)
            {
                names[key] = value;
            }
        }

        ChannelNames = names;
        MotionRatios = motionRatios ?? MotionRatios.Toyota86Zn6;
        ShockCalibration = shockCalibration;
        TrackWidths = trackWidths;
    }

    /// <summary>Human-readable vehicle name.</summary>
    public string Name { get; }

    /// <summary>Canonical key → actual logged channel name.</summary>
    public IReadOnlyDictionary<string, string> ChannelNames { get; }

    /// <summary>Pot/wheel motion ratios per corner.</summary>
    public MotionRatios MotionRatios { get; }

    /// <summary>Stroke-sensor calibration, when the profile defines it.</summary>
    public ShockCalibration? ShockCalibration { get; }

    /// <summary>Track widths for the roll gradient analysis, when known.</summary>
    public TrackWidths? TrackWidths { get; }

    /// <summary>The channel names for the given canonical keys (unknown keys skipped).</summary>
    public IReadOnlyDictionary<string, string> GetChannelSubset(IEnumerable<string> keys)
    {
        var subset = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var key in keys)
        {
            if (ChannelNames.TryGetValue(key, out var name))
            {
                subset[key] = name;
            }
        }

        return subset;
    }
}
