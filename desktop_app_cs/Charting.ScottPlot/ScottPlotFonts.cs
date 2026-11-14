using ScottPlot;
using SkiaSharp;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// Makes chart text use the UI font. A system family (Segoe UI on Windows)
/// is found by name; a font the app bundles (Inter on Linux) is not installed,
/// so the app hands its files over through <see cref="Register"/>.
/// </summary>
public static class ScottPlotFonts
{
    private static readonly BundledFontResolver Resolver = new();
    private static readonly object Gate = new();
    private static string _family = string.Empty;

    /// <summary>
    /// Offer a bundled font to ScottPlot under <paramref name="family"/>.
    /// The streams are read once, on first use.
    /// </summary>
    /// <param name="family">Family name as used in <see cref="ChartTheme.FontFamily"/>.</param>
    /// <param name="regular">Opens the regular-weight font file.</param>
    /// <param name="bold">Opens the bold (or semibold) font file; null reuses the regular one.</param>
    public static void Register(string family, Func<Stream> regular, Func<Stream>? bold = null)
    {
        lock (Gate)
        {
            Resolver.Add(family, regular, bold ?? regular);
            if (!Fonts.FontResolvers.Contains(Resolver))
            {
                Fonts.FontResolvers.Add(Resolver);
            }
        }
    }

    /// <summary>
    /// Make <paramref name="family"/> the default chart font when it can be
    /// resolved (registered or installed); otherwise ScottPlot's default
    /// stays. Empty leaves things as they are.
    /// </summary>
    internal static void UseFamily(string family)
    {
        if (family.Length == 0 || family == _family)
        {
            return;
        }

        lock (Gate)
        {
            _family = family;
            if (Resolver.Has(family) || IsInstalled(family))
            {
                Fonts.Default = family;
            }
        }
    }

    /// <summary>Whether the system font manager has <paramref name="family"/> itself (it substitutes a default otherwise).</summary>
    private static bool IsInstalled(string family)
    {
        using var typeface = SKFontManager.Default.MatchFamily(family);
        return typeface is not null && string.Equals(typeface.FamilyName, family, StringComparison.OrdinalIgnoreCase);
    }

    private sealed class BundledFontResolver : IFontResolver
    {
        private readonly Dictionary<string, (Func<Stream> Regular, Func<Stream> Bold)> _sources = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<(string, bool), SKTypeface?> _typefaces = new();

        public void Add(string family, Func<Stream> regular, Func<Stream> bold) => _sources[family] = (regular, bold);

        public bool Has(string family) => _sources.ContainsKey(family);

        public SKTypeface? CreateTypeface(string fontName, FontWeight weight, FontSlant slant, FontSpacing spacing) =>
            CreateTypeface(fontName, weight >= FontWeight.SemiBold, slant != FontSlant.Upright);

        public SKTypeface? CreateTypeface(string fontName, bool bold, bool italic)
        {
            lock (_typefaces)
            {
                if (!_sources.TryGetValue(fontName, out var source))
                {
                    return null;
                }

                if (!_typefaces.TryGetValue((fontName, bold), out var typeface))
                {
                    using var stream = (bold ? source.Bold : source.Regular)();
                    typeface = SKTypeface.FromStream(stream);
                    _typefaces[(fontName, bold)] = typeface;
                }

                return typeface;
            }
        }
    }
}
