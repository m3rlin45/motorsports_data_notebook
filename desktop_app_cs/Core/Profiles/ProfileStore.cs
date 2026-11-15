using System.Reflection;

namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Resolves vehicle profiles for a session: user profiles (the same
/// <c>profiles.yaml</c> the Python package writes) over the builtin ones
/// embedded from <c>builtin_profiles.yaml</c>. Port of
/// <c>get_profile_for_logger</c> / <c>get_profile_for_log</c> and friends.
/// </summary>
public sealed class ProfileStore
{
    private readonly Func<string?> _readUserYaml;
    private readonly Action<string> _writeUserYaml;
    private readonly Lazy<ProfilesYaml.Parsed> _builtin;

    /// <summary>A store over the builtin profiles and the user's config file.</summary>
    public ProfileStore()
        : this(UserProfilesPath())
    {
    }

    /// <summary>A store whose user profiles live at <paramref name="userProfilesPath"/>.</summary>
    public ProfileStore(string userProfilesPath)
        : this(
            () => File.Exists(userProfilesPath) ? File.ReadAllText(userProfilesPath) : null,
            text =>
            {
                Directory.CreateDirectory(Path.GetDirectoryName(userProfilesPath)!);
                File.WriteAllText(userProfilesPath, text);
            })
    {
        UserProfilesFile = userProfilesPath;
    }

    private ProfileStore(Func<string?> readUserYaml, Action<string> writeUserYaml)
    {
        _readUserYaml = readUserYaml;
        _writeUserYaml = writeUserYaml;
        _builtin = new Lazy<ProfilesYaml.Parsed>(() => ProfilesYaml.Parse(BuiltinYaml()));
    }

    /// <summary>Where user profiles are read from and saved to.</summary>
    public string? UserProfilesFile { get; }

    /// <summary>
    /// The user profiles file, as the Python package locates it:
    /// <c>$XDG_CONFIG_HOME/motorsports_data_notebook/profiles.yaml</c>
    /// (default <c>~/.config</c>) or <c>%APPDATA%\motorsports_data_notebook\profiles.yaml</c>.
    /// </summary>
    public static string UserProfilesPath()
    {
        string baseDir;
        if (OperatingSystem.IsWindows())
        {
            baseDir = Environment.GetEnvironmentVariable("APPDATA")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "AppData", "Roaming");
        }
        else
        {
            baseDir = Environment.GetEnvironmentVariable("XDG_CONFIG_HOME")
                ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".config");
        }

        return Path.Combine(baseDir, "motorsports_data_notebook", "profiles.yaml");
    }

    /// <summary>The embedded <c>builtin_profiles.yaml</c> text.</summary>
    public static string BuiltinYaml()
    {
        using var stream = typeof(ProfileStore).Assembly.GetManifestResourceStream("builtin_profiles.yaml")
            ?? throw new InvalidOperationException("builtin_profiles.yaml is not embedded");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    /// <summary>Builtin profiles keyed by <see cref="ProfileKey"/>.</summary>
    public IReadOnlyDictionary<string, VehicleProfile> BuiltinProfiles => _builtin.Value.Resolved;

    /// <summary>User profiles keyed by <see cref="ProfileKey"/>; empty when the file is missing or invalid.</summary>
    public IReadOnlyDictionary<string, VehicleProfile> LoadUserProfiles() => ParseUser().Resolved;

    /// <summary>All profile definitions keyed by id; user definitions override builtin ones.</summary>
    public IReadOnlyDictionary<string, VehicleProfile> LoadProfileDefinitions()
    {
        var definitions = new Dictionary<string, VehicleProfile>(_builtin.Value.Definitions, StringComparer.Ordinal);
        foreach (var (id, profile) in ParseUser().Definitions)
        {
            definitions[id] = profile;
        }

        return definitions;
    }

    /// <summary>Overwrite the user profiles file with <paramref name="profiles"/>.</summary>
    public void SaveUserProfiles(IReadOnlyDictionary<string, VehicleProfile> profiles) =>
        _writeUserYaml(ProfilesYaml.Write(profiles));

    /// <summary>Save one profile for a logger (and optional vehicle), merging with the existing user profiles.</summary>
    public void SaveProfileForLogger(string loggerId, VehicleProfile profile, string? vehicle = null)
    {
        var profiles = new Dictionary<string, VehicleProfile>(LoadUserProfiles(), StringComparer.Ordinal)
        {
            [ProfileKey.Make(loggerId, vehicle)] = profile,
        };
        SaveUserProfiles(profiles);
    }

    /// <summary>
    /// Look up a profile by logger ID and optional vehicle name. Order:
    /// vehicle-specific user, vehicle-specific builtin, logger-default user,
    /// logger-default builtin. Vehicle names match case-insensitively.
    /// </summary>
    public VehicleProfile? GetProfileForLogger(string loggerId, string? vehicle = null)
    {
        var sources = new[] { LoadUserProfiles(), BuiltinProfiles };

        if (!string.IsNullOrEmpty(vehicle))
        {
            var wanted = ProfileKey.Make(loggerId, vehicle);
            foreach (var source in sources)
            {
                foreach (var (key, profile) in source)
                {
                    if (string.Equals(key, wanted, StringComparison.OrdinalIgnoreCase))
                    {
                        return profile;
                    }
                }
            }
        }

        foreach (var source in sources)
        {
            if (source.TryGetValue(loggerId, out var profile))
            {
                return profile;
            }
        }

        return null;
    }

    /// <summary>Resolve the profile for a session from its metadata and file name.</summary>
    public VehicleProfile? GetProfileForLog(IReadOnlyDictionary<string, string> metadata, string? fileName = null)
    {
        var loggerId = LogIdentity.GetLoggerId(metadata);
        return string.IsNullOrEmpty(loggerId)
            ? null
            : GetProfileForLogger(loggerId, LogIdentity.GetVehicleName(metadata, fileName));
    }

    private ProfilesYaml.Parsed ParseUser()
    {
        try
        {
            var text = _readUserYaml();
            return text is null ? ProfilesYaml.Parsed.Empty : ProfilesYaml.Parse(text);
        }
        catch (Exception)
        {
            return ProfilesYaml.Parsed.Empty;
        }
    }
}
