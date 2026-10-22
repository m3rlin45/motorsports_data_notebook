using System.Globalization;
using YamlDotNet.RepresentationModel;

namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Parses and writes the two-tier profiles YAML (<c>profiles</c> definitions
/// plus a <c>logger_map</c>), following <c>_parse_profiles_yaml</c> /
/// <c>_profile_from_dict</c> / <c>save_user_profiles</c> in the Python
/// package. Invalid input yields empty results rather than exceptions, as in
/// Python. Uses YamlDotNet's representation model only, so it is AOT-safe.
/// </summary>
public static class ProfilesYaml
{
    /// <summary>Profile definitions keyed by id, and the logger map resolved to profiles.</summary>
    public sealed record Parsed(
        IReadOnlyDictionary<string, VehicleProfile> Definitions,
        IReadOnlyDictionary<string, VehicleProfile> Resolved)
    {
        /// <summary>The result of unreadable input.</summary>
        public static Parsed Empty { get; } = new(
            new Dictionary<string, VehicleProfile>(),
            new Dictionary<string, VehicleProfile>());
    }

    /// <summary>Parse YAML text. Unreadable or non-mapping input gives <see cref="Parsed.Empty"/>.</summary>
    public static Parsed Parse(string yamlText)
    {
        YamlMappingNode? root;
        try
        {
            var stream = new YamlStream();
            stream.Load(new StringReader(yamlText));
            root = stream.Documents.Count > 0 ? stream.Documents[0].RootNode as YamlMappingNode : null;
        }
        catch (Exception)
        {
            return Parsed.Empty;
        }

        if (root is null)
        {
            return Parsed.Empty;
        }

        var definitions = new Dictionary<string, VehicleProfile>(StringComparer.Ordinal);
        if (Child(root, "profiles") is YamlMappingNode profiles)
        {
            foreach (var (key, value) in profiles.Children)
            {
                if (key is YamlScalarNode { Value: { } id } && value is YamlMappingNode data)
                {
                    definitions[id] = ProfileFromNode(data);
                }
            }
        }

        var resolved = new Dictionary<string, VehicleProfile>(StringComparer.Ordinal);
        if (Child(root, "logger_map") is YamlMappingNode loggerMap)
        {
            foreach (var (key, entry) in loggerMap.Children)
            {
                if (key is not YamlScalarNode { Value: { } loggerId })
                {
                    continue;
                }

                switch (entry)
                {
                    case YamlScalarNode { Value: { } profileId } when definitions.TryGetValue(profileId, out var profile):
                        resolved[ProfileKey.Make(loggerId)] = profile;
                        break;
                    case YamlMappingNode nested:
                        if (Scalar(nested, "default") is { } defaultId && definitions.TryGetValue(defaultId, out var byDefault))
                        {
                            resolved[ProfileKey.Make(loggerId)] = byDefault;
                        }

                        if (Child(nested, "by_vehicle") is YamlMappingNode byVehicle)
                        {
                            foreach (var (vehicle, pid) in byVehicle.Children)
                            {
                                if (vehicle is YamlScalarNode { Value: { } vehicleName }
                                    && pid is YamlScalarNode { Value: { } vehicleProfileId }
                                    && definitions.TryGetValue(vehicleProfileId, out var vehicleProfile))
                                {
                                    resolved[ProfileKey.Make(loggerId, vehicleName)] = vehicleProfile;
                                }
                            }
                        }

                        break;
                }
            }
        }

        return new Parsed(definitions, resolved);
    }

    /// <summary>
    /// Build a profile from its mapping, as <c>_profile_from_dict</c> does:
    /// channel names on top of the defaults, motion ratios derived from the
    /// shock calibration (else the Toyota 86 values) with explicit entries
    /// overriding per corner, invalid calibration or track widths dropped.
    /// </summary>
    public static VehicleProfile ProfileFromNode(YamlMappingNode data)
    {
        Dictionary<string, string>? channelNames = null;
        if (Child(data, "channel_names") is YamlMappingNode names)
        {
            channelNames = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (key, value) in names.Children)
            {
                if (key is YamlScalarNode { Value: { } k } && value is YamlScalarNode v)
                {
                    channelNames[k] = v.Value ?? string.Empty;
                }
            }
        }

        ShockCalibration? calibration = null;
        if (Child(data, "shock_calibration") is YamlMappingNode cal)
        {
            var front = AxleFromNode(Child(cal, "front"));
            var rear = AxleFromNode(Child(cal, "rear"));
            if (front is not null && rear is not null)
            {
                calibration = new ShockCalibration(front, rear);
            }
        }

        var defaults = calibration?.PotWheelRatios() ?? MotionRatios.Toyota86Zn6;
        var ratiosNode = Child(data, "motion_ratios") as YamlMappingNode;
        var ratios = new MotionRatios(
            Number(ratiosNode, "front_left") ?? defaults.FrontLeft,
            Number(ratiosNode, "front_right") ?? defaults.FrontRight,
            Number(ratiosNode, "rear_left") ?? defaults.RearLeft,
            Number(ratiosNode, "rear_right") ?? defaults.RearRight);

        TrackWidths? trackWidths = null;
        if (Child(data, "track_width_mm") is YamlMappingNode tw
            && Number(tw, "front") is { } frontMm
            && Number(tw, "rear") is { } rearMm
            && frontMm > 0 && rearMm > 0)
        {
            trackWidths = new TrackWidths(frontMm, rearMm);
        }

        return new VehicleProfile(
            Scalar(data, "name") ?? "Unknown",
            channelNames,
            ratios,
            calibration,
            trackWidths);
    }

    /// <summary>Serialise profiles keyed by <see cref="ProfileKey"/> to the two-tier YAML text.</summary>
    public static string Write(IReadOnlyDictionary<string, VehicleProfile> profiles)
    {
        var definitions = new YamlMappingNode();
        var idByName = new Dictionary<string, string>(StringComparer.Ordinal);
        var loggerMap = new YamlMappingNode();

        foreach (var (key, profile) in profiles)
        {
            // A stable id from the name; the same profile under several loggers is written once.
            var id = profile.Name.ToLowerInvariant().Replace(' ', '_');
            if (!idByName.ContainsKey(profile.Name))
            {
                definitions.Add(id, ProfileToNode(profile));
                idByName[profile.Name] = id;
            }

            var pid = idByName[profile.Name];
            var (loggerId, vehicle) = ProfileKey.Split(key);
            var loggerKey = new YamlScalarNode(loggerId);
            loggerMap.Children.TryGetValue(loggerKey, out var existing);
            if (vehicle is null)
            {
                if (existing is YamlMappingNode nested)
                {
                    nested.Children[new YamlScalarNode("default")] = new YamlScalarNode(pid);
                }
                else
                {
                    loggerMap.Children[loggerKey] = new YamlScalarNode(pid);
                }
            }
            else
            {
                if (existing is not YamlMappingNode nested)
                {
                    nested = new YamlMappingNode();
                    if (existing is YamlScalarNode previousDefault)
                    {
                        nested.Add("default", previousDefault);
                    }

                    loggerMap.Children[loggerKey] = nested;
                }

                if (Child(nested, "by_vehicle") is not YamlMappingNode byVehicle)
                {
                    byVehicle = new YamlMappingNode();
                    nested.Add("by_vehicle", byVehicle);
                }

                byVehicle.Children[new YamlScalarNode(vehicle)] = new YamlScalarNode(pid);
            }
        }

        var root = new YamlMappingNode { { "profiles", definitions }, { "logger_map", loggerMap } };
        var writer = new StringWriter();
        new YamlStream(new YamlDocument(root)).Save(writer, assignAnchors: false);
        return writer.ToString();
    }

    private static YamlMappingNode ProfileToNode(VehicleProfile profile)
    {
        var channelNames = new YamlMappingNode();
        foreach (var (key, value) in profile.ChannelNames)
        {
            channelNames.Add(key, value);
        }

        var node = new YamlMappingNode
        {
            { "name", profile.Name },
            { "channel_names", channelNames },
            {
                "motion_ratios", new YamlMappingNode
                {
                    { "front_left", Num(profile.MotionRatios.FrontLeft) },
                    { "front_right", Num(profile.MotionRatios.FrontRight) },
                    { "rear_left", Num(profile.MotionRatios.RearLeft) },
                    { "rear_right", Num(profile.MotionRatios.RearRight) },
                }
            },
        };

        if (profile.ShockCalibration is { } cal)
        {
            node.Add("shock_calibration", new YamlMappingNode
            {
                { "front", AxleToNode(cal.Front) },
                { "rear", AxleToNode(cal.Rear) },
            });
        }

        if (profile.TrackWidths is { } tw)
        {
            node.Add("track_width_mm", new YamlMappingNode
            {
                { "front", Num(tw.FrontMm) },
                { "rear", Num(tw.RearMm) },
            });
        }

        return node;
    }

    private static YamlMappingNode AxleToNode(AxleShockCalibration axle)
    {
        var node = new YamlMappingNode
        {
            { "wheel_per_sensor", Num(axle.WheelPerSensor) },
            { "damper_per_sensor", Num(axle.DamperPerSensor) },
        };
        if (axle.DamperPerWheelFit is { } dpw)
        {
            node.Add("damper_per_wheel", Num(dpw));
        }

        if (axle.WheelR2 is { } r2w)
        {
            node.Add("wheel_r2", Num(r2w));
        }

        if (axle.DamperR2 is { } r2d)
        {
            node.Add("damper_r2", Num(r2d));
        }

        if (axle.DamperWheelR2 is { } r2dw)
        {
            node.Add("damper_wheel_r2", Num(r2dw));
        }

        if (axle.Points is { Count: > 0 } points)
        {
            var rows = new YamlSequenceNode();
            foreach (var row in points)
            {
                var seq = new YamlSequenceNode { Style = YamlDotNet.Core.Events.SequenceStyle.Flow };
                foreach (var v in row)
                {
                    seq.Add(Num(v));
                }

                rows.Add(seq);
            }

            node.Add("points", rows);
        }

        if (axle.ExcludedPoints is { Count: > 0 } excluded)
        {
            var seq = new YamlSequenceNode { Style = YamlDotNet.Core.Events.SequenceStyle.Flow };
            foreach (var i in excluded)
            {
                seq.Add(i.ToString(CultureInfo.InvariantCulture));
            }

            node.Add("excluded_points", seq);
        }

        return node;
    }

    private static AxleShockCalibration? AxleFromNode(YamlNode? node)
    {
        if (node is not YamlMappingNode axle
            || Number(axle, "wheel_per_sensor") is not { } wheelPerSensor
            || Number(axle, "damper_per_sensor") is not { } damperPerSensor)
        {
            return null;
        }

        List<IReadOnlyList<double>>? points = null;
        if (Child(axle, "points") is YamlSequenceNode rows)
        {
            points = new List<IReadOnlyList<double>>();
            foreach (var row in rows)
            {
                if (row is not YamlSequenceNode values)
                {
                    return null;
                }

                var parsed = new List<double>();
                foreach (var v in values)
                {
                    if (ParseNumber(v) is not { } d)
                    {
                        return null;
                    }

                    parsed.Add(d);
                }

                points.Add(parsed);
            }
        }

        List<int>? excluded = null;
        if (Child(axle, "excluded_points") is YamlSequenceNode indices)
        {
            excluded = new List<int>();
            foreach (var v in indices)
            {
                if (v is not YamlScalarNode { Value: { } text }
                    || !int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i))
                {
                    return null;
                }

                excluded.Add(i);
            }
        }

        return new AxleShockCalibration(
            wheelPerSensor,
            damperPerSensor,
            Number(axle, "damper_per_wheel"),
            Number(axle, "wheel_r2"),
            Number(axle, "damper_r2"),
            Number(axle, "damper_wheel_r2"),
            points,
            excluded);
    }

    private static YamlNode? Child(YamlMappingNode node, string key) =>
        node.Children.TryGetValue(new YamlScalarNode(key), out var child) ? child : null;

    private static string? Scalar(YamlMappingNode node, string key) =>
        Child(node, key) is YamlScalarNode { Value: { } value } ? value : null;

    private static double? Number(YamlMappingNode? node, string key) =>
        node is null ? null : ParseNumber(Child(node, key));

    private static double? ParseNumber(YamlNode? node) =>
        node is YamlScalarNode { Value: { } text }
        && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : null;

    private static YamlScalarNode Num(double value) => new(value.ToString("R", CultureInfo.InvariantCulture));
}
