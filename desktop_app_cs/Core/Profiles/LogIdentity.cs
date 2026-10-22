namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Reads the logger ID and vehicle name a session is identified by. Port of
/// <c>get_logger_id</c> / <c>get_vehicle_name</c>; metadata values are the
/// string form of what the parser reports.
/// </summary>
public static class LogIdentity
{
    private static readonly string[] VehicleKeys = { "Vehicle", "Vehicle Name", "vehicle_name", "vehicle" };

    /// <summary>
    /// The logger serial for AIM files, <c>"iracing"</c> for IBT files (all
    /// iRacing cars share channel names), else null.
    /// </summary>
    public static string? GetLoggerId(IReadOnlyDictionary<string, string> metadata)
    {
        if (LoggerIdValue(metadata) is { } id)
        {
            return id;
        }

        return metadata.ContainsKey("session_info_yaml") ? "iracing" : null;
    }

    /// <summary>True when the metadata is from an iRacing IBT file.</summary>
    public static bool IsIracingSession(IReadOnlyDictionary<string, string> metadata) =>
        GetLoggerId(metadata) == "iracing";

    /// <summary>
    /// The vehicle name: the <c>Vehicle</c> metadata when set (Race Studio
    /// writes it to .xrk files), else the vehicle token of the Race Studio
    /// file name for AIM files (.xrz exports leave the field empty).
    /// </summary>
    public static string? GetVehicleName(IReadOnlyDictionary<string, string> metadata, string? fileName = null)
    {
        foreach (var key in VehicleKeys)
        {
            if (metadata.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value))
            {
                return value.Trim();
            }
        }

        // Only AIM files follow the Race Studio naming scheme. Python tests the
        // key for `is None` here (an empty value still counts), unlike GetLoggerId.
        if (!(metadata.ContainsKey("Logger ID") || metadata.ContainsKey("logger_id")) || string.IsNullOrEmpty(fileName))
        {
            return null;
        }

        return VehicleFromFileName(fileName);
    }

    /// <summary>
    /// The vehicle token of a Race Studio 3 file name
    /// <c>&lt;Racer&gt;_&lt;Vehicle&gt;_&lt;Track&gt;_&lt;Session&gt;_a_&lt;NNNN&gt;</c>, or null.
    /// </summary>
    public static string? VehicleFromFileName(string fileName)
    {
        var stem = Path.GetFileName(fileName);
        var dot = stem.LastIndexOf('.');
        if (dot >= 0)
        {
            stem = stem[..dot];
        }

        var parts = stem.Split('_');
        if (parts.Length < 3)
        {
            return null;
        }

        var vehicle = parts[1].Trim();
        return vehicle.Length == 0 ? null : vehicle;
    }

    // Python: `metadata.get("Logger ID") or metadata.get("logger_id")` — empty strings fall through.
    private static string? LoggerIdValue(IReadOnlyDictionary<string, string> metadata)
    {
        foreach (var key in new[] { "Logger ID", "logger_id" })
        {
            if (metadata.TryGetValue(key, out var value) && !string.IsNullOrEmpty(value))
            {
                return value;
            }
        }

        return null;
    }
}
