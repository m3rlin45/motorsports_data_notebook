using System.Text.Json;
using System.Text.Json.Serialization;

namespace InfernoAnalyzer.App;

/// <summary>
/// Lap data dumped by <c>tools/extract_fixture.py</c>. Stands in for the real
/// data layer (libxrk FFI, Phase 1) during the charting spike.
/// </summary>
public sealed class Fixture
{
    /// <summary>Source file the laps came from.</summary>
    [JsonPropertyName("source")]
    public string Source { get; set; } = string.Empty;

    /// <summary>Timebase channel the series were resampled onto.</summary>
    [JsonPropertyName("timebase")]
    public string Timebase { get; set; } = string.Empty;

    /// <summary>Channel name → unit string.</summary>
    [JsonPropertyName("units")]
    public Dictionary<string, string> Units { get; set; } = new();

    /// <summary>Channels in display order.</summary>
    [JsonPropertyName("channels")]
    public List<string> Channels { get; set; } = new();

    /// <summary>Laps in the order they should be drawn; the first is the reference lap.</summary>
    [JsonPropertyName("laps")]
    public List<FixtureLap> Laps { get; set; } = new();

    /// <summary>Total number of samples across all laps and channels.</summary>
    [JsonIgnore]
    public int SampleCount => Laps.Sum(lap => lap.DistanceM.Length * (Channels.Count + 1));

    /// <summary>Read a fixture from a JSON file.</summary>
    public static Fixture Load(string path)
    {
        using var stream = File.OpenRead(path);
        return JsonSerializer.Deserialize(stream, FixtureJsonContext.Default.Fixture)
            ?? throw new InvalidDataException($"Empty fixture: {path}");
    }
}

/// <summary>One lap of the fixture. All arrays are the same length.</summary>
public sealed class FixtureLap
{
    /// <summary>Lap number as logged.</summary>
    [JsonPropertyName("num")]
    public int Num { get; set; }

    /// <summary>Lap time in seconds.</summary>
    [JsonPropertyName("lap_time_s")]
    public double LapTimeS { get; set; }

    /// <summary>Lap-relative time of each sample, seconds.</summary>
    [JsonPropertyName("time_s")]
    public double[] TimeS { get; set; } = Array.Empty<double>();

    /// <summary>Cumulative distance of each sample, metres.</summary>
    [JsonPropertyName("distance_m")]
    public double[] DistanceM { get; set; } = Array.Empty<double>();

    /// <summary>Channel name → samples.</summary>
    [JsonPropertyName("series")]
    public Dictionary<string, double[]> Series { get; set; } = new();

    /// <summary>Legend label.</summary>
    [JsonIgnore]
    public string Label => $"Lap {Num} ({LapTimeS:0.000}s)";
}

/// <summary>Source-generated JSON context so the app stays NativeAOT/trim friendly.</summary>
[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(Fixture))]
internal sealed partial class FixtureJsonContext : JsonSerializerContext
{
}
