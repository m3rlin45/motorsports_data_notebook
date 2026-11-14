using System.Text.Json;
using System.Text.Json.Serialization;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.Settings;

/// <summary>
/// What the app remembers between runs: where the window was, what was open,
/// and how each tab was set up. Stored as JSON beside the vehicle profiles.
/// Every field has a usable default, so a missing or older file just means
/// starting from the defaults.
/// </summary>
public sealed class AppSettings
{
    /// <summary>Most entries kept in <see cref="RecentFiles"/>.</summary>
    public const int MaxRecentFiles = 10;

    /// <summary>Window width when not maximized (device-independent pixels).</summary>
    public double WindowWidth { get; set; } = 1280;

    /// <summary>Window height when not maximized.</summary>
    public double WindowHeight { get; set; } = 800;

    /// <summary>Whether the window was maximized.</summary>
    public bool Maximized { get; set; }

    /// <summary>Width of the session sidebar.</summary>
    public double SidebarWidth { get; set; } = 288;

    /// <summary>Whether the session sidebar is hidden.</summary>
    public bool SidebarCollapsed { get; set; }

    /// <summary>"System", "Light" or "Dark".</summary>
    public string Theme { get; set; } = "System";

    /// <summary>
    /// Windows: put the menu in the title bar instead of under it. On by
    /// default; View > Menu in Title Bar turns it off.
    /// </summary>
    public bool MenuInTitleBar { get; set; } = true;

    /// <summary>Header of the tab that was showing.</summary>
    public string? LastTab { get; set; }

    /// <summary>Folder the last session was opened from.</summary>
    public string? LastFolder { get; set; }

    /// <summary>Recently opened session files, newest first.</summary>
    public List<string> RecentFiles { get; set; } = new();

    /// <summary>Per-tab view state, keyed by tab header then by the tab's own keys.</summary>
    public Dictionary<string, Dictionary<string, string>> Tabs { get; set; } = new();

    /// <summary>
    /// Channels ticked on the Time/Distance tab, per vehicle profile (keyed by
    /// logger ID and vehicle name), so each car comes back with its own set.
    /// </summary>
    public Dictionary<string, List<string>> ChannelsByVehicle { get; set; } = new();

    /// <summary>Put <paramref name="path"/> at the head of <see cref="RecentFiles"/> and remember its folder.</summary>
    public void AddRecent(string path)
    {
        RecentFiles.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        RecentFiles.Insert(0, path);
        if (RecentFiles.Count > MaxRecentFiles)
        {
            RecentFiles.RemoveRange(MaxRecentFiles, RecentFiles.Count - MaxRecentFiles);
        }

        LastFolder = Path.GetDirectoryName(path) ?? LastFolder;
    }
}

/// <summary>Loads and saves <see cref="AppSettings"/>. A store without a path keeps the settings in memory only (tests, demos).</summary>
public sealed class SettingsStore
{
    private readonly string? _path;

    /// <summary>A store over <paramref name="path"/>; null keeps everything in memory.</summary>
    public SettingsStore(string? path = null)
    {
        _path = path;
        Settings = Load();
    }

    /// <summary>The settings file beside the user's vehicle profiles.</summary>
    public static string DefaultPath() =>
        Path.Combine(Path.GetDirectoryName(ProfileStore.UserProfilesPath())!, "analyzer_settings.json");

    /// <summary>The current settings; edit in place, then <see cref="Save"/>.</summary>
    public AppSettings Settings { get; }

    private AppSettings Load()
    {
        if (_path is null || !File.Exists(_path))
        {
            return new AppSettings();
        }

        try
        {
            using var stream = File.OpenRead(_path);
            return JsonSerializer.Deserialize(stream, SettingsJsonContext.Default.AppSettings) ?? new AppSettings();
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return new AppSettings(); // an unreadable settings file must never stop the app from starting
        }
    }

    /// <summary>Write the settings; failures are swallowed (losing preferences is better than failing to close).</summary>
    public void Save()
    {
        if (_path is null)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            var temp = _path + ".tmp";
            using (var stream = File.Create(temp))
            {
                JsonSerializer.Serialize(stream, Settings, SettingsJsonContext.Default.AppSettings);
            }

            File.Move(temp, _path, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }
}

/// <summary>Source-generated JSON context so the app stays NativeAOT/trim friendly.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(AppSettings))]
internal sealed partial class SettingsJsonContext : JsonSerializerContext
{
}
