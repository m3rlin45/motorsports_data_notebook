using System.Collections.Generic;
using System.Globalization;
using TirePressureCalculator.Localization;
using Xunit;

namespace TirePressureCalculator.Tests;

public class LocalizationTests
{
    private static readonly string[] RequiredKeys =
    {
        "LanguageLabel", "LanguageAuto", "LanguageEnglish", "LanguageJapanese",
        "ModeLabel", "ModeManual", "ModePrediction", "ModelUnavailable",
        "CurrentTemp", "TargetTempCorr", "PredictedHotTemp", "TargetBar",
        "SetCold", "ResetButton",
        "Track", "Car", "Condition", "LapWithinStint", "Ambient", "CloudCover",
        "ConditionDry", "ConditionDamp", "ConditionWet",
        "TempAdjustZero", "TempAdjustFormat",
        "PredictedHotPrefix",
    };

    [Fact]
    public void Localizer_HasEnglishStringsForAllRequiredKeys()
    {
        Localizer.Instance.SetPreference("en");
        foreach (var key in RequiredKeys)
        {
            var value = Localizer.Instance[key];
            Assert.False(string.IsNullOrWhiteSpace(value), $"Missing English value for key '{key}'");
        }
    }

    [Fact]
    public void Localizer_HasJapaneseStringsForAllRequiredKeys()
    {
        Localizer.Instance.SetPreference("ja");
        foreach (var key in RequiredKeys)
        {
            var value = Localizer.Instance[key];
            Assert.False(string.IsNullOrWhiteSpace(value), $"Missing Japanese value for key '{key}'");
        }
        Localizer.Instance.SetPreference("en");
    }

    [Fact]
    public void Localizer_JapaneseDiffersFromEnglish_ForUserFacingStrings()
    {
        Localizer.Instance.SetPreference("en");
        var en = new Dictionary<string, string>();
        foreach (var key in RequiredKeys) en[key] = Localizer.Instance[key];

        Localizer.Instance.SetPreference("ja");
        // At minimum, mode labels and the main button must be translated
        // (proving the JA lookup actually swaps strings, not just falls back).
        Assert.NotEqual(en["ModeManual"], Localizer.Instance["ModeManual"]);
        Assert.NotEqual(en["ResetButton"], Localizer.Instance["ResetButton"]);
        Assert.NotEqual(en["Track"], Localizer.Instance["Track"]);
        Localizer.Instance.SetPreference("en");
    }

    [Fact]
    public void Localizer_UnknownKeyReturnsKeyItself()
    {
        Localizer.Instance.SetPreference("en");
        Assert.Equal("ThisKeyDoesNotExist", Localizer.Instance["ThisKeyDoesNotExist"]);
    }

    [Fact]
    public void Localizer_AutoResolvesToBrowserCultureOrFallsBackToEnglish()
    {
        // Simulate a Japanese browser
        var original = CultureInfo.CurrentUICulture;
        try
        {
            CultureInfo.CurrentUICulture = new CultureInfo("ja-JP");
            Localizer.Instance.SetPreference("auto");
            Assert.Equal("ja", Localizer.Instance.Language);

            // Unsupported culture falls back to English
            CultureInfo.CurrentUICulture = new CultureInfo("fr-FR");
            Localizer.Instance.SetPreference("auto");
            Assert.Equal("en", Localizer.Instance.Language);
        }
        finally
        {
            CultureInfo.CurrentUICulture = original;
            Localizer.Instance.SetPreference("en");
        }
    }

    [Fact]
    public void Localizer_FormatInjectsArgumentIntoTemplate()
    {
        Localizer.Instance.SetPreference("en");
        var formatted = Localizer.Instance.Format("TempAdjustFormat", "+1.5");
        Assert.Contains("+1.5", formatted);
        Assert.DoesNotContain("{0}", formatted);
    }

    [Fact]
    public void Localizer_LanguageChangeFiresPropertyChanged()
    {
        Localizer.Instance.SetPreference("en");
        bool fired = false;
        Localizer.Instance.PropertyChanged += handler;
        try
        {
            Localizer.Instance.SetPreference("ja");
            Assert.True(fired,
                "Expected PropertyChanged(Language) so view models can refresh their T_* bindings.");
        }
        finally
        {
            Localizer.Instance.PropertyChanged -= handler;
            Localizer.Instance.SetPreference("en");
        }

        void handler(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(Localizer.Language)) fired = true;
        }
    }
}
