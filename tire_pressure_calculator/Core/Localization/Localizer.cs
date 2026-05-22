using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TirePressureCalculator.Localization;

/// <summary>
/// Bundled-JSON localizer.
///
/// Strings live in an embedded resource (<c>strings.json</c>), keyed by
/// two-letter language code (<c>en</c>, <c>ja</c>). Switching language fires
/// <c>PropertyChanged("Item[]")</c> so all bindings created via <see cref="LocalizeExtension"/>
/// refresh in place — no XAML reload required.
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

    /// <summary>
    /// Two-letter language code currently in effect (already resolved — never "auto").
    /// </summary>
    public string Language
    {
        get => _language;
        private set
        {
            if (_language == value) return;
            _language = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Language)));
            // "Item[]" tells WPF/Avalonia to invalidate every indexer binding,
            // but Avalonia's reflection-based binding plugin sometimes only
            // listens for the specific indexer path it's bound to. Fire both
            // the generic invalidation and per-key notifications so corner
            // labels (which bind via {l:Localize KEY} → Source[KEY]) refresh
            // live on language switch.
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Item[]"));
            if (_strings.TryGetValue(_language, out var current))
            {
                foreach (var key in current.Keys)
                {
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs($"Item[{key}]"));
                }
            }
        }
    }

    /// <summary>
    /// Apply a user-facing preference: "auto", "en", "ja", or any other supported code.
    /// "auto" resolves via <see cref="CultureInfo.CurrentUICulture"/>; unknown values fall
    /// back to English.
    /// </summary>
    public void SetPreference(string preference)
    {
        Language = ResolvePreference(preference);
    }

    private static string ResolvePreference(string preference)
    {
        if (string.IsNullOrWhiteSpace(preference) || preference == "auto")
        {
            var cultureCode = CultureInfo.CurrentUICulture.TwoLetterISOLanguageName;
            return IsSupported(cultureCode) ? cultureCode : FallbackLanguage;
        }
        return IsSupported(preference) ? preference : FallbackLanguage;
    }

    private static bool IsSupported(string code)
    {
        foreach (var s in SupportedLanguages)
            if (s == code) return true;
        return false;
    }

    public static IReadOnlyList<string> AvailableLanguages => SupportedLanguages;

    /// <summary>
    /// Look up a string by key. Falls back to English, then to the key itself
    /// (so missing keys are visible in the UI but don't crash).
    /// </summary>
    public string this[string key]
    {
        get
        {
            if (_strings.TryGetValue(_language, out var lang)
                && lang.TryGetValue(key, out var value))
            {
                return value;
            }
            if (_strings.TryGetValue(FallbackLanguage, out var en)
                && en.TryGetValue(key, out var enValue))
            {
                return enValue;
            }
            return key;
        }
    }

    public string Format(string key, params object[] args)
    {
        return string.Format(this[key], args);
    }

    private static Dictionary<string, Dictionary<string, string>> LoadStrings()
    {
        var asm = typeof(Localizer).Assembly;
        using var stream = asm.GetManifestResourceStream("strings.json")
            ?? throw new InvalidOperationException(
                "Embedded resource 'strings.json' not found. "
                + "Check TirePressureCalculator.Core.csproj <EmbeddedResource>.");
        using var reader = new StreamReader(stream);
        var json = reader.ReadToEnd();
        return JsonSerializer.Deserialize(json, LocalizationJsonContext.Default.DictionaryStringDictionaryStringString)
            ?? new Dictionary<string, Dictionary<string, string>>();
    }

    public event PropertyChangedEventHandler? PropertyChanged;
}

[JsonSerializable(typeof(Dictionary<string, Dictionary<string, string>>))]
internal partial class LocalizationJsonContext : JsonSerializerContext { }
