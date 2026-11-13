using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Suspension velocity analysis against Python on the Fuji 86 sample
/// (Fixtures/suspension_86.json). The inputs come from the C# loader (the
/// 500 Hz shock channels are too large to dump), so histogram shares are
/// compared with a one-sample allowance per bin and moments to 1e-6.
/// </summary>
public class SuspensionParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "suspension_86.json")))!;

    private static readonly Lazy<LogFile> Session = new(() =>
        SessionLoader.Load(SessionParityTests.SamplePath(), new ProfileStore("/nonexistent/profiles.yaml")));

    private static Dictionary<string, string> ChannelNames() =>
        Fixture["channels"]!.AsObject().ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());

    private static MotionRatios Ratios()
    {
        var mr = Fixture["motion_ratios"]!;
        return new MotionRatios(mr["front_left"]!.GetValue<double>(), mr["front_right"]!.GetValue<double>(), mr["rear_left"]!.GetValue<double>(), mr["rear_right"]!.GetValue<double>());
    }

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    [Fact]
    public void Velocity_derivation_matches_python_exactly()
    {
        var slice = Fixture["velocity_slice"]!;
        var displacement = Doubles(slice["displacement"]!);
        var timecodes = slice["timecodes"]!.AsArray().Select(v => v!.GetValue<long>()).ToArray();

        Assert.Equal(Doubles(slice["raw_velocity"]!), Suspension.ComputeShockVelocity(displacement, timecodes, smoothingWindow: 1));
        var smoothed = Suspension.ComputeShockVelocity(displacement, timecodes, smoothingWindow: 5);
        var expected = Doubles(slice["smoothed_velocity"]!);
        Assert.Equal(expected.Length, smoothed.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], smoothed[i], Math.Abs(expected[i]) * 1e-12 + 1e-12);
        }
    }

    public static TheoryData<string> Cases()
    {
        var data = new TheoryData<string>();
        foreach (var lap in Fixture["single_lap"]!.AsObject())
        {
            data.Add(lap.Key);
        }

        data.Add("multi");
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Histograms_and_stats_match_python(string which)
    {
        var names = ChannelNames();
        JsonNode expected;
        VelocityHistogramResult actual;
        if (which == "multi")
        {
            expected = Fixture["multi_lap"]!;
            var laps = Fixture["single_lap"]!.AsObject().Select(p => int.Parse(p.Key)).ToList();
            actual = Suspension.AnalyzeSuspensionVelocityMultiLap(Session.Value, laps, names, Ratios(), cancellation: TestContext.Current.CancellationToken);
        }
        else
        {
            expected = Fixture["single_lap"]![which]!;
            actual = Suspension.AnalyzeSuspensionVelocity(Session.Value.FilterByLap(int.Parse(which)), names, Ratios());
        }

        foreach (var (key, corner) in new[] { ("front_left", actual.FrontLeft), ("front_right", actual.FrontRight), ("rear_left", actual.RearLeft), ("rear_right", actual.RearRight) })
        {
            var e = expected[key]!;
            Assert.Equal(e["corner_name"]!.GetValue<string>(), corner.CornerName);
            Assert.Equal(Doubles(e["bin_edges"]!), corner.BinEdges);
            Assert.Equal(Doubles(e["bin_centers"]!), corner.BinCenters);

            var n = e["velocity"]!["length"]!.GetValue<int>();
            Assert.Equal(n, corner.Velocity.Length);
            var oneSample = 100.0 / n;
            var hist = Doubles(e["histogram"]!);
            Assert.Equal(hist.Length, corner.Histogram.Length);
            for (var i = 0; i < hist.Length; i++)
            {
                Assert.True(Math.Abs(hist[i] - corner.Histogram[i]) <= oneSample + 1e-9, $"{which} {key} bin {i}: {hist[i]} vs {corner.Histogram[i]}");
            }

            foreach (var (stat, value) in new[] { ("mean", corner.Mean), ("std", corner.Std), ("skew", corner.Skew), ("kurtosis", corner.Kurtosis) })
            {
                var ev = e[stat]!.GetValue<double>();
                Assert.True(Math.Abs(ev - value) <= Math.Max(1e-6, Math.Abs(ev) * 1e-6), $"{which} {key} {stat}: {ev} vs {value}");
            }

            foreach (var (stat, value) in new[] { ("pct_friction", corner.PctFriction), ("pct_slow_bump", corner.PctSlowBump), ("pct_slow_rebound", corner.PctSlowRebound), ("pct_fast_bump", corner.PctFastBump), ("pct_fast_rebound", corner.PctFastRebound), ("pct_curb", corner.PctCurb), ("pct_zero_bin", corner.PctZeroBin) })
            {
                var ev = e[stat]!.GetValue<double>();
                Assert.True(Math.Abs(ev - value) <= 2 * oneSample + 1e-9, $"{which} {key} {stat}: {ev} vs {value}");
            }
        }

        var symmetry = expected["symmetry"]!.AsArray();
        var corners = actual.Corners.ToList();
        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(symmetry[i]!["Bump/Total %"]!.GetValue<double>(), corners[i].BumpOfTotalPct, 1e-3);
        }

        var comparison = expected["comparison"]!.AsArray();
        var (leftVsRight, frontVsRear) = actual.GroupDifferences();
        Assert.Equal(comparison[0]!["Curb Diff"]!.GetValue<double>(), leftVsRight.Curb, 1e-3);
        Assert.Equal(comparison[1]!["Friction Diff"]!.GetValue<double>(), frontVsRear.Friction, 1e-3);
    }
}

/// <summary>Exact-math unit tests for the suspension primitives.</summary>
public class SuspensionTests
{
    [Fact]
    public void Gradient_with_coordinates_matches_numpy_formula()
    {
        // numpy.gradient([0, 1, 4, 9], [0, 1, 3, 4]) = [1, 7/6, 23/6, 5]
        double[] f = { 0, 1, 4, 9 };
        double[] x = { 0, 1, 3, 4 };

        var g = Numerics.Gradient(f, x);

        Assert.Equal(1.0, g[0]);
        Assert.Equal(7.0 / 6.0, g[1], 1e-12);
        Assert.Equal(23.0 / 6.0, g[2], 1e-12);
        Assert.Equal(5.0, g[3]);
    }

    [Fact]
    public void Histogram_bins_are_zero_centred_and_sum_to_100()
    {
        var velocity = new double[] { -400, -7, -3, 0, 2, 6, 14, 500 };

        var (hist, edges, centers) = Suspension.ComputeVelocityHistogram(velocity, binSize: 10, maxVelocity: 300);

        Assert.Equal(62, edges.Length);
        Assert.Equal(-305, edges[0]);
        Assert.Equal(305, edges[^1]);
        Assert.Equal(0, centers[30]);
        Assert.Equal(100.0, hist.Sum(), 1e-9);
        Assert.Equal(100.0 * 3 / 8, hist[30]); // -5 <= v < 5 holds -3, 0, 2
        Assert.Equal(100.0 / 8, hist[0]); // -400 clipped to -300 lands in the first bin
        Assert.Equal(100.0 / 8, hist[^1]); // 500 clipped to 300 lands in the last bin
    }

    [Fact]
    public void Stats_categorise_by_absolute_velocity_and_sign()
    {
        var velocity = new double[] { 0, 3, -10, 20, -100, 150, 250, -300 };

        var s = Suspension.ComputeVelocityStats(velocity, new VelocityRanges());

        Assert.Equal(25.0, s.PctFriction);
        Assert.Equal(12.5, s.PctSlowBump);
        Assert.Equal(12.5, s.PctSlowRebound);
        Assert.Equal(12.5, s.PctFastBump);
        Assert.Equal(12.5, s.PctFastRebound);
        Assert.Equal(25.0, s.PctCurb);
        Assert.Equal(velocity.Average(), s.Mean, 1e-12);
    }

    [Fact]
    public void Smoothing_pads_edges()
    {
        double[] displacement = { 0, 1, 3, 6, 10, 15, 21 };
        long[] timecodes = { 0, 100, 200, 300, 400, 500, 600 };

        var raw = Suspension.ComputeShockVelocity(displacement, timecodes, smoothingWindow: 1);
        var smooth = Suspension.ComputeShockVelocity(displacement, timecodes, smoothingWindow: 3);

        Assert.Equal(raw.Length, smooth.Length);
        Assert.Equal((raw[0] + raw[0] + raw[1]) / 3, smooth[0], 1e-9);
        Assert.Equal((raw[1] + raw[2] + raw[3]) / 3, smooth[2], 1e-9);
    }
}
