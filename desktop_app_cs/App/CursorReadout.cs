using System.Globalization;
using System.Text;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App;

/// <summary>
/// Formats the per-lap channel values under the cursor, e.g.
/// <c>2158 m | Speed: 123.6 / 124.8 | Throttle: 100.0 / 100.0</c>.
/// </summary>
public static class CursorReadout
{
    /// <summary>Text for cursor position <paramref name="x"/> (null when the cursor is hidden).</summary>
    public static string Format(Fixture fixture, XAxisMode xAxis, double? x)
    {
        if (x is null)
        {
            return "Hover the chart for lap values";
        }

        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append(xAxis == XAxisMode.Distance
            ? string.Format(inv, "{0:0} m", x.Value)
            : string.Format(inv, "{0:0.00} s", x.Value));

        foreach (var channel in fixture.Channels)
        {
            sb.Append(" | ").Append(channel).Append(": ");
            var first = true;
            foreach (var lap in fixture.Laps)
            {
                if (!lap.Series.TryGetValue(channel, out var ys))
                {
                    continue;
                }

                var xs = xAxis == XAxisMode.Distance ? lap.DistanceM : lap.TimeS;
                var value = StripChartBuilder.Interpolate(xs, ys, new[] { x.Value })[0];
                if (!first)
                {
                    sb.Append(" / ");
                }

                sb.Append(value.ToString("0.0", inv));
                first = false;
            }
        }

        return sb.ToString();
    }
}
