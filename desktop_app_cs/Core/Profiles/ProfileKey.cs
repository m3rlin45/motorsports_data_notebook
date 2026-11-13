namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Lookup keys for the logger map: <c>"&lt;logger id&gt;"</c> or
/// <c>"&lt;logger id&gt;/&lt;vehicle&gt;"</c> for a logger shared between cars.
/// </summary>
public static class ProfileKey
{
    /// <summary>Separator between logger ID and vehicle name.</summary>
    public const char Separator = '/';

    /// <summary>Build the key for a logger and optional vehicle name.</summary>
    public static string Make(string loggerId, string? vehicle = null)
    {
        loggerId = loggerId.Trim();
        return string.IsNullOrEmpty(vehicle) ? loggerId : $"{loggerId}{Separator}{vehicle.Trim()}";
    }

    /// <summary>Inverse of <see cref="Make"/>.</summary>
    public static (string LoggerId, string? Vehicle) Split(string key)
    {
        var i = key.IndexOf(Separator);
        if (i < 0)
        {
            return (key, null);
        }

        var vehicle = key[(i + 1)..];
        return (key[..i], vehicle.Length == 0 ? null : vehicle);
    }
}
