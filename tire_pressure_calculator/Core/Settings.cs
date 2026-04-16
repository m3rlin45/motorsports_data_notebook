using System.Text.Json;
using System.Text.Json.Serialization;

namespace TirePressureCalculator;

public class CornerSettings
{
    public double CurrentTemp { get; set; } = 20.0;
    public double TargetHotTemp { get; set; } = 80.0;
    public double TargetHotPressure { get; set; } = 1.80;
}

public class AppSettings
{
    public CornerSettings FrontLeft { get; set; } = new();
    public CornerSettings FrontRight { get; set; } = new();
    public CornerSettings RearLeft { get; set; } = new();
    public CornerSettings RearRight { get; set; } = new();
    public double TempAdjustPercent { get; set; }

    private static string FilePath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TirePressureCalculator",
            "settings.json");

    public static AppSettings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                var json = File.ReadAllText(FilePath);
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
            var dir = Path.GetDirectoryName(FilePath)!;
            Directory.CreateDirectory(dir);
            var json = JsonSerializer.Serialize(this, SettingsContext.Default.AppSettings);
            File.WriteAllText(FilePath, json);
        }
        catch { }
    }
}

// AOT-compatible JSON serialization
[JsonSerializable(typeof(AppSettings))]
[JsonSourceGenerationOptions(WriteIndented = true)]
internal partial class SettingsContext : JsonSerializerContext { }
