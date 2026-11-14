using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

/// <summary>The chart theme: token resolution and the legibility of the lap palette on each surface.</summary>
public class ChartThemeTests
{
    private static double Luminance(string hex)
    {
        static double Channel(int c)
        {
            var v = c / 255.0;
            return v <= 0.03928 ? v / 12.92 : Math.Pow((v + 0.055) / 1.055, 2.4);
        }

        var r = Convert.ToInt32(hex.Substring(1, 2), 16);
        var g = Convert.ToInt32(hex.Substring(3, 2), 16);
        var b = Convert.ToInt32(hex.Substring(5, 2), 16);
        return 0.2126 * Channel(r) + 0.7152 * Channel(g) + 0.0722 * Channel(b);
    }

    /// <summary>WCAG contrast ratio of two colours.</summary>
    private static double Contrast(string a, string b)
    {
        var (hi, lo) = (Math.Max(Luminance(a), Luminance(b)), Math.Min(Luminance(a), Luminance(b)));
        return (hi + 0.05) / (lo + 0.05);
    }

    [Fact]
    public void Tokens_resolve_per_theme_and_literals_pass_through()
    {
        Assert.Equal("#123456", ChartTheme.Light.Resolve("#123456"));
        Assert.NotEqual(ChartTheme.Light.Resolve(ChartColors.SessionA), ChartTheme.Dark.Resolve(ChartColors.SessionA));
        Assert.Equal(ChartTheme.Light.Text, ChartTheme.Light.Resolve(ChartColors.Ink));
        Assert.Equal(ChartTheme.Dark.Text, ChartTheme.Dark.Resolve(ChartColors.Ink));
        // An unknown token must still draw: it falls back to the text colour rather than throwing mid-render.
        Assert.Equal(ChartTheme.Dark.Text, ChartTheme.Dark.Resolve("@no-such-token"));
        Assert.Equal(ChartColors.Lap(0), ChartColors.Lap(ChartColors.LapSlots)); // wraps
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Every_series_colour_reads_against_the_data_area(bool dark)
    {
        var theme = dark ? ChartTheme.Dark : ChartTheme.Light;
        var tokens = Enumerable.Range(0, ChartColors.LapSlots).Select(ChartColors.Lap)
            .Concat(new[] { ChartColors.SessionA, ChartColors.SessionB, ChartColors.Bump, ChartColors.Rebound, ChartColors.Braking, ChartColors.Acceleration, ChartColors.Warning, ChartColors.Opportunity });
        foreach (var token in tokens)
        {
            var ratio = Contrast(theme.Resolve(token), theme.DataBackground);
            Assert.True(ratio >= 3.0, $"{token} = {theme.Resolve(token)} is {ratio:0.00}:1 on {theme.DataBackground}");
        }

        Assert.True(Contrast(theme.Text, theme.DataBackground) >= 7.0);
        Assert.True(Contrast(theme.Axis, theme.FigureBackground) >= 4.5);
    }

    [Fact]
    public void High_contrast_themes_use_pure_ink_and_heavier_lines()
    {
        Assert.True(ChartTheme.HighContrastDark.HighContrast);
        Assert.Equal("#ffffff", ChartTheme.HighContrastDark.Resolve(ChartColors.Ink));
        Assert.Equal("#000000", ChartTheme.HighContrastLight.Resolve(ChartColors.Neutral));
        Assert.True(ChartTheme.HighContrastLight.LineScale > ChartTheme.Light.LineScale);
    }
}
