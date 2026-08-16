using Avalonia;
using Avalonia.Media;

namespace TirePressureCalculator;

/// <summary>
/// Configures Avalonia's font manager with the embedded Noto Sans JP subset
/// as a fallback, so Japanese characters render on hosts without a suitable
/// system font in Avalonia's default fallback chain.
/// </summary>
public static class FontConfiguration
{
    /// <summary>avares:// URI of the bundled Japanese font subset.</summary>
    public const string JapaneseFontUri =
        "avares://TirePressureCalculator.Core/Assets/Fonts/NotoSansJP-Subset.ttf#Noto Sans JP";

    /// <summary>Apply the JA-aware font fallback chain to an AppBuilder.</summary>
    public static AppBuilder WithJapaneseFontFallback(this AppBuilder builder) =>
        builder.With(new FontManagerOptions
        {
            FontFallbacks = new[]
            {
                new FontFallback { FontFamily = new FontFamily(JapaneseFontUri) },
            },
        });
}
