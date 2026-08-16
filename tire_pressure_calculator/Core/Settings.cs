using System.Text.Json;
using System.Text.Json.Serialization;

namespace TirePressureCalculator;

public class CornerSettings
{
    public double CurrentTemp { get; set; } = 20.0;
    public double TargetHotTemp { get; set; } = 80.0;
    public double TargetHotPressure { get; set; } = 1.80;
}

// Platform-pluggable persistence: Desktop/Android use the on-disk default.
// (The standalone web app at ../web mirrors this JSON shape in localStorage.)
public interface ISettingsStorage
{
    string? Read();
    void Write(string contents);
}

public enum AppMode
{
    Manual = 0,
    CircuitPrediction = 1,
}

public class PredictionSettings
{
    public string? Track { get; set; }
    public string? Car { get; set; }
    public string Condition { get; set; } = "dry";
    public int LapWithinStint { get; set; } = 5;
    public double AmbientTempC { get; set; } = 20.0;
    public double? TrackTempC { get; set; }
    public double? CloudCoverPct { get; set; }
    public double? TargetLapTimeS { get; set; }
}

public class AppSettings
{
    public CornerSettings FrontLeft { get; set; } = new();
    public CornerSettings FrontRight { get; set; } = new();
    public CornerSettings RearLeft { get; set; } = new();
    public CornerSettings RearRight { get; set; } = new();
    public double TempAdjustPercent { get; set; }
    public AppMode Mode { get; set; } = AppMode.Manual;
    public PredictionSettings Prediction { get; set; } = new();
    /// <summary>UI language preference: "auto", "en", or "ja".</summary>
    public string UiLanguage { get; set; } = "auto";

    public static ISettingsStorage Storage { get; set; } = new FileSettingsStorage();

    public static AppSettings Load()
    {
        try
        {
            var json = Storage.Read();
            if (!string.IsNullOrEmpty(json))
            {
                return JsonSerializer.Deserialize(json, SettingsContext.Default.AppSettings) ?? new();
            }
        }
        catch { }
        return new();
    }

    public void Save()
    {
        try
        {
            var json = JsonSerializer.Serialize(this, SettingsContext.Default.AppSettings);
            Storage.Write(json);
        }
        catch { }
    }
}

public class FileSettingsStorage : ISettingsStorage
{
    private static string FilePath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TirePressureCalculator",
            "settings.json");

    public string? Read() => File.Exists(FilePath) ? File.ReadAllText(FilePath) : null;

    public void Write(string contents)
    {
        var dir = Path.GetDirectoryName(FilePath)!;
        Directory.CreateDirectory(dir);
        File.WriteAllText(FilePath, contents);
    }
}

// AOT-compatible JSON serialization
[JsonSerializable(typeof(AppSettings))]
[JsonSourceGenerationOptions(WriteIndented = true)]
internal partial class SettingsContext : JsonSerializerContext { }
