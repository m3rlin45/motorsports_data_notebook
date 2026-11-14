using System.Globalization;
using YamlDotNet.RepresentationModel;

namespace InfernoAnalyzer.Core.Tracks;

/// <summary>A WGS-84 position.</summary>
public readonly record struct GeoPoint(double Lat, double Lon);

/// <summary>
/// A sector gate: a chord across the track between two GPS points. A lap
/// crosses it when the GPS path intersects the chord.
/// </summary>
public sealed record SectorGate(string Name, GeoPoint A, GeoPoint B);

/// <summary>
/// A track: an id, a display name, an anchor position for matching sessions
/// to it, and the sector gates in lap order. Start/finish is implicit (the
/// logger's lap boundaries); sector 1 runs from the lap start to gate 1,
/// the last sector from the last gate to the lap end.
/// </summary>
public sealed record TrackDefinition(string Id, string Name, GeoPoint Anchor, IReadOnlyList<SectorGate> Gates)
{
    /// <summary>Number of sectors: gates + 1.</summary>
    public int SectorCount => Gates.Count + 1;

    /// <summary>"S1", "S2", … or the gate names when they are set.</summary>
    public IReadOnlyList<string> SectorNames =>
        Enumerable.Range(1, SectorCount).Select(i => $"S{i}").ToList();
}

/// <summary>Local tangent-plane projection shared by gates and GPS paths.</summary>
public static class GeoProjection
{
    private const double EarthRadius = 6371000;
    private const double DegToRad = Math.PI / 180.0;

    /// <summary>Metres east/north of <paramref name="origin"/> (equirectangular, like <c>gps_to_local_xy</c>).</summary>
    public static (double X, double Y) ToLocal(GeoPoint p, GeoPoint origin)
    {
        var x = EarthRadius * ((p.Lon - origin.Lon) * DegToRad) * Math.Cos(origin.Lat * DegToRad);
        var y = EarthRadius * ((p.Lat - origin.Lat) * DegToRad);
        return (x, y);
    }

    /// <summary>Inverse of <see cref="ToLocal"/>.</summary>
    public static GeoPoint FromLocal(double x, double y, GeoPoint origin)
    {
        var lat = origin.Lat + y / EarthRadius / DegToRad;
        var lon = origin.Lon + x / EarthRadius / DegToRad / Math.Cos(origin.Lat * DegToRad);
        return new GeoPoint(lat, lon);
    }

    /// <summary>Great-circle distance in metres (haversine).</summary>
    public static double DistanceM(GeoPoint a, GeoPoint b)
    {
        var dLat = (b.Lat - a.Lat) * DegToRad;
        var dLon = (b.Lon - a.Lon) * DegToRad;
        var h = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) + Math.Cos(a.Lat * DegToRad) * Math.Cos(b.Lat * DegToRad) * Math.Sin(dLon / 2) * Math.Sin(dLon / 2);
        return 2 * EarthRadius * Math.Asin(Math.Sqrt(h));
    }
}

/// <summary>Reads and writes the tracks YAML (<c>tracks:</c> list with id, name, anchor and gates).</summary>
public static class TrackYaml
{
    /// <summary>Parse YAML text; unreadable input gives an empty list.</summary>
    public static List<TrackDefinition> Parse(string yamlText)
    {
        var tracks = new List<TrackDefinition>();
        YamlMappingNode? root;
        try
        {
            var stream = new YamlStream();
            stream.Load(new StringReader(yamlText));
            root = stream.Documents.Count > 0 ? stream.Documents[0].RootNode as YamlMappingNode : null;
        }
        catch (Exception)
        {
            return tracks;
        }

        if (root is null || Child(root, "tracks") is not YamlSequenceNode list)
        {
            return tracks;
        }

        foreach (var item in list)
        {
            if (item is not YamlMappingNode t || Scalar(t, "id") is not { } id)
            {
                continue;
            }

            var gates = new List<SectorGate>();
            if (Child(t, "gates") is YamlSequenceNode gateList)
            {
                foreach (var g in gateList)
                {
                    if (g is YamlMappingNode gate && Point(gate, "a") is { } a && Point(gate, "b") is { } b)
                    {
                        gates.Add(new SectorGate(Scalar(gate, "name") ?? $"Gate {gates.Count + 1}", a, b));
                    }
                }
            }

            var anchor = Point(t, "anchor") ?? (gates.Count > 0 ? gates[0].A : new GeoPoint(0, 0));
            tracks.Add(new TrackDefinition(id, Scalar(t, "name") ?? id, anchor, gates));
        }

        return tracks;
    }

    /// <summary>Serialise tracks to YAML text.</summary>
    public static string Write(IEnumerable<TrackDefinition> tracks)
    {
        var list = new YamlSequenceNode();
        foreach (var t in tracks)
        {
            var gates = new YamlSequenceNode();
            foreach (var g in t.Gates)
            {
                gates.Add(new YamlMappingNode
                {
                    { "name", g.Name },
                    { "a", PointNode(g.A) },
                    { "b", PointNode(g.B) },
                });
            }

            list.Add(new YamlMappingNode
            {
                { "id", t.Id },
                { "name", t.Name },
                { "anchor", PointNode(t.Anchor) },
                { "gates", gates },
            });
        }

        var writer = new StringWriter();
        new YamlStream(new YamlDocument(new YamlMappingNode { { "tracks", list } })).Save(writer, assignAnchors: false);
        return writer.ToString();
    }

    private static YamlMappingNode PointNode(GeoPoint p) => new()
    {
        { "lat", p.Lat.ToString("R", CultureInfo.InvariantCulture) },
        { "lon", p.Lon.ToString("R", CultureInfo.InvariantCulture) },
    };

    private static GeoPoint? Point(YamlMappingNode node, string key)
    {
        if (Child(node, key) is not YamlMappingNode p)
        {
            return null;
        }

        return Number(p, "lat") is { } lat && Number(p, "lon") is { } lon ? new GeoPoint(lat, lon) : null;
    }

    private static YamlNode? Child(YamlMappingNode node, string key) =>
        node.Children.TryGetValue(new YamlScalarNode(key), out var child) ? child : null;

    private static string? Scalar(YamlMappingNode node, string key) =>
        Child(node, key) is YamlScalarNode { Value: { } value } ? value : null;

    private static double? Number(YamlMappingNode node, string key) =>
        Scalar(node, key) is { } text && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : null;
}

/// <summary>
/// Tracks shipped with the app (embedded <c>builtin_tracks.yaml</c>) plus the
/// user's <c>tracks.yaml</c> next to their profiles. User tracks override
/// builtin ones with the same id.
/// </summary>
public sealed class TrackStore
{
    private readonly string _userPath;

    /// <summary>A store over the user's tracks file in the app's config directory.</summary>
    public TrackStore()
        : this(Path.Combine(Path.GetDirectoryName(Profiles.ProfileStore.UserProfilesPath())!, "tracks.yaml"))
    {
    }

    /// <summary>A store whose user tracks live at <paramref name="userPath"/>.</summary>
    public TrackStore(string userPath)
    {
        _userPath = userPath;
    }

    /// <summary>Where user tracks are saved.</summary>
    public string UserTracksFile => _userPath;

    /// <summary>Builtin tracks from the embedded YAML (empty when none are embedded).</summary>
    public static IReadOnlyList<TrackDefinition> Builtin()
    {
        using var stream = typeof(TrackStore).Assembly.GetManifestResourceStream("builtin_tracks.yaml");
        if (stream is null)
        {
            return Array.Empty<TrackDefinition>();
        }

        using var reader = new StreamReader(stream);
        return TrackYaml.Parse(reader.ReadToEnd());
    }

    /// <summary>User tracks; empty when the file is missing or unreadable.</summary>
    public List<TrackDefinition> LoadUser()
    {
        try
        {
            return File.Exists(_userPath) ? TrackYaml.Parse(File.ReadAllText(_userPath)) : new List<TrackDefinition>();
        }
        catch (Exception)
        {
            return new List<TrackDefinition>();
        }
    }

    /// <summary>All tracks, user definitions overriding builtin ids.</summary>
    public List<TrackDefinition> LoadAll()
    {
        var byId = Builtin().ToDictionary(t => t.Id, StringComparer.Ordinal);
        foreach (var t in LoadUser())
        {
            byId[t.Id] = t;
        }

        return byId.Values.OrderBy(t => t.Name, StringComparer.OrdinalIgnoreCase).ToList();
    }

    /// <summary>Add or replace a user track.</summary>
    public void Save(TrackDefinition track)
    {
        var tracks = LoadUser().Where(t => t.Id != track.Id).Append(track).ToList();
        Directory.CreateDirectory(Path.GetDirectoryName(_userPath)!);
        File.WriteAllText(_userPath, TrackYaml.Write(tracks));
    }

    /// <summary>Remove a user track (builtin tracks cannot be removed).</summary>
    public void Delete(string id)
    {
        var tracks = LoadUser().Where(t => t.Id != id).ToList();
        Directory.CreateDirectory(Path.GetDirectoryName(_userPath)!);
        File.WriteAllText(_userPath, TrackYaml.Write(tracks));
    }

    /// <summary>The track whose anchor is nearest to <paramref name="position"/> within <paramref name="maxDistanceM"/>, or null.</summary>
    public TrackDefinition? FindNear(GeoPoint position, double maxDistanceM = 3000)
    {
        return LoadAll()
            .Select(t => (Track: t, Distance: GeoProjection.DistanceM(t.Anchor, position)))
            .Where(x => x.Distance <= maxDistanceM)
            .OrderBy(x => x.Distance)
            .Select(x => x.Track)
            .FirstOrDefault();
    }

    /// <summary>A stable id from a name: lower case, runs of non-alphanumerics collapsed to one underscore.</summary>
    public static string IdFor(string name)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var c in name.Trim().ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(c);
            }
            else if (sb.Length > 0 && sb[^1] != '_')
            {
                sb.Append('_');
            }
        }

        var id = sb.ToString().Trim('_');
        return id.Length == 0 ? "track" : id;
    }
}
