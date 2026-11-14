using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>Roll gradient against Python on the Fuji 86 sample (Fixtures/roll_gradient_86.json), inputs from the C# loader.</summary>
public class RollGradientParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "roll_gradient_86.json")))!;

    private static readonly Lazy<LogFile> Session = new(() =>
        SessionLoader.Load(SessionParityTests.SamplePath(), new ProfileStore("/nonexistent/profiles.yaml")));

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    private static Dictionary<string, string> ChannelNames() =>
        Fixture["channels"]!.AsObject().ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());

    private static MotionRatios Ratios()
    {
        var mr = Fixture["motion_ratios"]!;
        return new MotionRatios(mr["front_left"]!.GetValue<double>(), mr["front_right"]!.GetValue<double>(), mr["rear_left"]!.GetValue<double>(), mr["rear_right"]!.GetValue<double>());
    }

    private static TrackWidths Widths() =>
        new(Fixture["track_widths"]!["front_mm"]!.GetValue<double>(), Fixture["track_widths"]!["rear_mm"]!.GetValue<double>());

    private static void AssertClose(double expected, double actual, double relative, string what)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"{what}: expected NaN, got {actual}");
            return;
        }

        Assert.True(Math.Abs(expected - actual) <= Math.Max(1e-9, Math.Abs(expected) * relative), $"{what}: expected {expected}, got {actual}");
    }

    [Fact]
    public void Fit_and_binning_match_python_on_synthetic_data()
    {
        var syn = Fixture["synthetic"]!;
        var g = Doubles(syn["lateral_g"]!);
        var roll = Doubles(syn["roll_deg"]!);

        var (gradient, intercept, r2) = RollGradient.FitRollGradient(g, roll, minLateralG: 0.2);
        var fit = Doubles(syn["fit"]!);
        AssertClose(fit[0], gradient, 1e-10, "gradient");
        AssertClose(fit[1], intercept, 1e-10, "intercept");
        AssertClose(fit[2], r2, 1e-10, "r2");

        var (centers, means, counts) = RollGradient.BinRollByLateralG(g, roll, 0.25, 3);
        Assert.Equal(Doubles(syn["bin_centers"]!), centers);
        Assert.Equal(syn["bin_counts"]!.AsArray().Select(v => v!.GetValue<long>()), counts);
        var expectedMeans = Doubles(syn["bin_means"]!);
        for (var i = 0; i < expectedMeans.Length; i++)
        {
            AssertClose(expectedMeans[i], means[i], 1e-12, $"bin mean {i}");
        }

        var smoothed = RollGradient.Smooth(roll[..12], 4);
        var expectedSmoothed = Doubles(syn["smoothed_even"]!);
        Assert.Equal(expectedSmoothed.Length, smoothed.Length);
        for (var i = 0; i < smoothed.Length; i++)
        {
            AssertClose(expectedSmoothed[i], smoothed[i], 1e-12, $"smoothed {i}");
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
    public void Analysis_matches_python(string which)
    {
        JsonNode expected;
        RollGradientResult actual;
        if (which == "multi")
        {
            expected = Fixture["multi_lap"]!;
            var laps = Fixture["single_lap"]!.AsObject().Select(p => int.Parse(p.Key)).ToList();
            actual = RollGradient.AnalyzeRollGradientMultiLap(Session.Value, laps, ChannelNames(), Ratios(), Widths(), cancellation: TestContext.Current.CancellationToken);
        }
        else
        {
            expected = Fixture["single_lap"]![which]!;
            actual = RollGradient.AnalyzeRollGradient(Session.Value.FilterByLap(int.Parse(which)), ChannelNames(), Ratios(), Widths());
        }

        Assert.Equal(expected["lateral_g_unit"]!.GetValue<string>(), actual.LateralGUnit);
        foreach (var (key, axle) in new[] { ("front", actual.Front), ("rear", actual.Rear) })
        {
            var e = expected[key]!;
            Assert.Equal(e["n_samples"]!.GetValue<int>(), axle.NSamples);
            Assert.Equal(e["lateral_g"]!["length"]!.GetValue<int>(), axle.LateralG.Length);
            AssertClose(e["lateral_g"]!["sum"]!.GetValue<double>(), Numerics.Sum(axle.LateralG), 1e-6, $"{key} lateral sum");
            AssertClose(e["roll_deg"]!["sum"]!.GetValue<double>(), Numerics.Sum(axle.RollDeg), 1e-6, $"{key} roll sum");
            AssertClose(e["gradient_deg_per_g"]!.GetValue<double>(), axle.GradientDegPerG, 1e-6, $"{key} gradient");
            AssertClose(e["intercept_deg"]!.GetValue<double>(), axle.InterceptDeg, 1e-6, $"{key} intercept");
            AssertClose(e["r2"]!.GetValue<double>(), axle.R2, 1e-6, $"{key} r2");

            var centers = Doubles(e["bin_centers_g"]!);
            Assert.Equal(centers.Length, axle.BinCentersG.Length);
            var means = Doubles(e["bin_mean_deg"]!);
            var counts = e["bin_counts"]!.AsArray().Select(v => v!.GetValue<long>()).ToArray();
            for (var i = 0; i < centers.Length; i++)
            {
                AssertClose(centers[i], axle.BinCentersG[i], 1e-9, $"{key} centre {i}");
                // Smoothed lateral-G samples sit exactly on 0.1 g bin edges often enough that
                // last-bit summation order moves a handful of them between neighbouring bins.
                Assert.True(Math.Abs(counts[i] - axle.BinCounts[i]) <= Math.Max(10, counts[i] / 100), $"{key} bin {i} count {counts[i]} vs {axle.BinCounts[i]}");
                AssertClose(means[i], axle.BinMeanDeg[i], 1e-3, $"{key} bin mean {i}");
            }
        }

        AssertClose(expected["roll_ratio"]!.GetValue<double>(), actual.RollRatio, 1e-6, "roll ratio");
        AssertClose(expected["roll_ratio_r2"]!.GetValue<double>(), actual.RollRatioR2, 1e-6, "roll ratio r2");
        AssertClose(expected["gradient_ratio"]!.GetValue<double>(), actual.GradientRatio, 1e-6, "gradient ratio");
        AssertClose(expected["mean_gradient_deg_per_g"]!.GetValue<double>(), actual.MeanGradientDegPerG, 1e-6, "mean gradient");
        Assert.Equal(expected["signs_consistent"]!.GetValue<bool>(), actual.SignsConsistent);
        Assert.Equal(Doubles(expected["ratio_bin_centers_deg"]!).Length, actual.RatioBinCentersDeg.Length);

        Assert.Equal(expected["x_m"] is not null, actual.HasTrackMap);
        if (actual.HasTrackMap)
        {
            Assert.Equal(expected["x_m"]!["length"]!.GetValue<int>(), actual.XM!.Length);
            AssertClose(expected["x_m"]!["max"]!.GetValue<double>(), actual.XM!.Max(), 1e-6, "x max");
        }
    }
}

/// <summary>Unit tests for the roll primitives.</summary>
public class RollGradientTests
{
    [Fact]
    public void Roll_angle_is_atan_of_travel_difference_over_track()
    {
        var roll = RollGradient.ComputeRollAngleDeg(new double[] { 10 }, new double[] { -10 }, 1500);

        Assert.Equal(0.7639, roll[0], 1e-4);
        Assert.Throws<ArgumentOutOfRangeException>(() => RollGradient.ComputeRollAngleDeg(new double[] { 1 }, new double[] { 1 }, 0));
    }

    [Fact]
    public void Fit_is_nan_without_spread()
    {
        var (g, i, r2) = RollGradient.FitRollGradient(new double[] { 0.5, 0.5, 0.5 }, new double[] { 1, 2, 3 });

        Assert.True(double.IsNaN(g) && double.IsNaN(i) && double.IsNaN(r2));
    }

    [Fact]
    public void Lateral_in_ms2_is_converted_to_g()
    {
        Assert.Equal(new[] { 1.0 }, RollGradient.LateralToG(new[] { 9.80665 }, "m/s^2"));
        Assert.Equal(new[] { 1.0 }, RollGradient.LateralToG(new[] { 1.0 }, "g"));
    }
}
