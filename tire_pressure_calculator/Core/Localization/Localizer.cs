using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TirePressureCalculator.Localization;

/// <summary>
/// Bundled-JSON localizer. Strings live in an embedded <c>strings.json</c>
/// keyed by two-letter language code (<c>en</c>, <c>ja</c>). View models
/// re-expose each label they need as a <c>T_*</c> property and subscribe
/// to <see cref="PropertyChanged"/> here so a language switch refreshes
/// every binding without a XAML reload.
/// </summary>
public sealed class Localizer : INotifyPropertyChanged
{
    public static Localizer Instance { get; } = new();

    private const string FallbackLanguage = "en";
    private static readonly string[] SupportedLanguages = { "en", "ja" };

    private readonly Dictionary<string, Dictionary<string, string>> _strings;
    private string _language = FallbackLanguage;

    private Localizer()
    {
        _strings = LoadStrings();
    }

    /// <summary>Two-letter code currently in effect — never "auto".</summary>
    public string Language
    {
        get => _language;
        private set
        {
            if (_language == value) return;
            _language = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Language)));
        }
    }

    /// <summary>
    /// Apply a user-facing preference: <c>"auto"</c>, <c>"en"</c>, or
    /// <c>"ja"</c>. <c>"auto"</c> resolves via
    /// <see cref="CultureInfo.CurrentUICulture"/>; unknown values fall back
    /// to English.
    /// </summary>
    public void SetPreference(string preference) => Language = Resolve(preference);

    public static IReadOnlyList<string> AvailableLanguages => SupportedLanguages;

    /// <summary>
    /// Look up a string by key. Falls back to English, then to the key
    /// itself (so a missing key is visible in the UI without crashing).
    /// </summary>
    public string this[string key]
    {
        get
        {
            if (_strings.TryGetValue(_language, out var lang)
                && lang.TryGetValue(key, out var value)) return value;
            if (_strings.TryGetValue(FallbackLanguage, out var en)
                && en.TryGetValue(key, out var enValue)) return enValue;
            return key;
        }
    }

    public string Format(string key, params object[] args) => string.Format(this[key], args);

    public event PropertyChangedEventHandler? PropertyChanged;

    private static string Resolve(string preference)
    {
        if (string.IsNullOrWhiteSpace(preference) || preference == "auto")
            preference = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName;
        foreach (var s in SupportedLanguages)
            if (s == preference) return preference;
        return FallbackLanguage;
    }

    private static Dictionary<string, Dictionary<string, string>> LoadStrings()
    {
        using var stream = typeof(Localizer).Assembly.GetManifestResourceStream("strings.json")
            ?? throw new InvalidOperationException(
                "Embedded resource 'strings.json' not found. "
                + "Check TirePressureCalculator.Core.csproj <EmbeddedResource>.");
        using var reader = new StreamReader(stream);
        return JsonSerializer.Deserialize(
            reader.ReadToEnd(),
            LocalizationJsonContext.Default.DictionaryStringDictionaryStringString)
            ?? new Dictionary<string, Dictionary<string, string>>();
    }
}

[JsonSerializable(typeof(Dictionary<string, Dictionary<string, string>>))]
internal partial class LocalizationJsonContext : JsonSerializerContext { }
