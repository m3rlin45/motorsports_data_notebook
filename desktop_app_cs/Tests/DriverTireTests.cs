using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>Throttle acceptance against Python on the Fuji 86 sample (Fixtures/driver_analysis_86.json).</summary>
public class DriverAnalysisParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "driver_analysis_86.json")))!;

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    private static List<Corner> CornersFromFixture() =>
        Fixture["corners"]!.AsArray().Select(c => new Corner
        {
            Id = c!["id"]!.GetValue<int>(),
            Name = c["name"]!.GetValue<string>(),
            Direction = c["direction"]!.GetValue<string>()[0],
            StartDist = c["start_dist"]!.GetValue<double>(),
            EndDist = c["end_dist"]!.GetValue<double>(),
            ApexDist = c["apex_dist"]!.GetValue<double>(),
            MaxCurvature = c["max_curvature"]!.GetValue<double>(),
        }).ToList();

    [Fact]
    public void Throttle_acceptance_matches_python_per_corner_and_lap()
    {
        var corners = CornersFromFixture();
        foreach (var (lapNum, lapNode) in Fixture["laps"]!.AsObject())
        {
            var inputs = lapNode!["inputs"]!;
            var distance = Doubles(inputs["distance"]!);
            var timecodes = inputs["timecodes"]!.AsArray().Select(v => v!.GetValue<long>()).ToArray();
            var throttle = Doubles(inputs["throttle"]!);
            var lateral = Doubles(inputs["lateral_g"]!);

            var (smoothed, effective) = DriverAnalysis.PrepareThrottleAcceptance(throttle, lateral, 98.0);
            Assert.Equal(lapNode["effective_threshold"]!.GetValue<double>(), effective, 1e-9);
            var sm = lapNode["smoothed_lateral_g"]!;
            Assert.Equal(sm["sum"]!.GetValue<double>(), Numerics.Sum(smoothed), Math.Abs(sm["sum"]!.GetValue<double>()) * 1e-9);

            var expected = lapNode["throttle_acceptance"]!.AsArray();
            Assert.Equal(expected.Count, corners.Count);
            for (var i = 0; i < corners.Count; i++)
            {
                var ta = DriverAnalysis.FindThrottleAcceptance(distance, timecodes, throttle, lateral, corners[i], smoothed, effective);
                var e = expected[i];
                if (e is null)
                {
                    Assert.True(ta is null, $"lap {lapNum} corner {corners[i].Name}: expected no acceptance, got {ta}");
                    continue;
                }

                Assert.True(ta is not null, $"lap {lapNum} corner {corners[i].Name}: expected acceptance, got null");
                Assert.Equal(e["throttle_acceptance_pct"]!.GetValue<double>(), ta!.ThrottleAcceptancePct, 1e-6);
                Assert.Equal(e["lateral_g_at_throttle"]!.GetValue<double>(), ta.LateralGAtThrottle, 1e-9);
                Assert.Equal(e["peak_lateral_g"]!.GetValue<double>(), ta.PeakLateralG, 1e-9);
                Assert.Equal(e["full_throttle_dist"]!.GetValue<double>(), ta.FullThrottleDist, 1e-9);
            }
        }
    }

    /// <summary>The whole desktop pipeline runs on the loaded session and agrees with the fixture's throttle-acceptance values.</summary>
    [Fact]
    public void Driver_consistency_pipeline_runs_on_the_session()
    {
        var store = new ProfileStore("/nonexistent/profiles.yaml");
        var session = SessionLoader.Load(SessionParityTests.SamplePath(), store);
        var profile = store.GetProfileForLog(session.Metadata, session.FileName)!;
        var laps = Fixture["laps"]!.AsObject().Select(p => int.Parse(p.Key)).ToList();

        var result = DriverConsistency.Analyze(session, laps, profile.ChannelNames);

        Assert.Equal(CornersFromFixture().Count, result.Corners.Count);
        Assert.Equal(result.Corners.Count * 3, result.Segments.Count);
        Assert.Equal(result.Corners.Count * 3 * laps.Count, result.Stats.Count);
        for (var ci = 0; ci < result.Corners.Count; ci++)
        {
            // The pipeline visits laps in table order (ascending lap number), the fixture in request order.
            var expectedTa = Fixture["laps"]!.AsObject()
                .OrderBy(p => int.Parse(p.Key))
                .Select(p => p.Value!["throttle_acceptance"]![ci])
                .Where(e => e is not null)
                .Select(e => e!["throttle_acceptance_pct"]!.GetValue<double>())
                .ToList();
            var cd = result.CornerData[ci];
            Assert.Equal(expectedTa.Count, cd.TaValues.Count);
            for (var i = 0; i < expectedTa.Count; i++)
            {
                Assert.Equal(expectedTa[i], cd.TaValues[i], 1e-6);
            }

            Assert.Equal(laps.Count, cd.LapTraces.Count);
            Assert.True(cd.SpeedValues.Count > 0);
        }
    }
}

/// <summary>
/// Tire grip against Python on the Fuji 86 sample (Fixtures/tire_grip_86.json).
/// The lateral, inline and TPMS channels are float32 arrays in Python, so
/// numpy computes the means, bucket edges and percentiles in single
/// precision; the C# port computes in double, hence 1e-6 tolerances and a
/// small allowance for samples that sit on a bucket edge.
/// </summary>
public class TireGripParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tire_grip_86.json")))!;

    private static readonly Lazy<LogFile> Session = new(() =>
        SessionLoader.Load(SessionParityTests.SamplePath(), new ProfileStore("/nonexistent/profiles.yaml")));

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    private static Dictionary<string, string> ChannelNames() =>
        Fixture["channels"]!.AsObject().ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());

    private static void AssertCorner(JsonNode e, CornerTireGripData c, double relative)
    {
        Assert.Equal(e["corner_name"]!.GetValue<string>(), c.CornerName);
        Assert.Equal(e["total_g"]!["length"]!.GetValue<int>(), c.TotalG.Length);
        foreach (var (key, actual) in new[] { ("mean_g", c.MeanG), ("mean_metric", c.MeanMetric), ("std_g", c.StdG), ("std_metric", c.StdMetric) })
        {
            var expected = e[key]!.GetValue<double>();
            Assert.True(Math.Abs(expected - actual) <= Math.Max(1e-9, Math.Abs(expected) * relative), $"{c.CornerName} {key}: {expected} vs {actual}");
        }

        var centers = Doubles(e["bucket_centers"]!);
        var values = Doubles(e["bucket_values"]!);
        var counts = e["bucket_counts"]!.AsArray().Select(v => v!.GetValue<long>()).ToArray();
        Assert.Equal(centers.Length, c.BucketCenters.Length);
        for (var i = 0; i < centers.Length; i++)
        {
            Assert.Equal(centers[i], c.BucketCenters[i], Math.Abs(centers[i]) * relative);
            Assert.True(Math.Abs(counts[i] - c.BucketCounts[i]) <= Math.Max(2, counts[i] / 100), $"{c.CornerName} bucket {i} count {counts[i]} vs {c.BucketCounts[i]}");
            Assert.True(Math.Abs(values[i] - c.BucketValues[i]) <= Math.Max(1e-6, Math.Abs(values[i]) * 1e-3), $"{c.CornerName} bucket {i} value {values[i]} vs {c.BucketValues[i]}");
        }
    }

    [Fact]
    public void Bucketed_percentile_matches_python_on_synthetic_data()
    {
        var syn = Fixture["synthetic"]!;
        var corner = TireGrip.ComputeCorner(Doubles(syn["total_g"]!), Doubles(syn["metric"]!), "FL");

        AssertCorner(syn["corner"]!, corner, 1e-12);
        Assert.Equal(Doubles(syn["corner"]!["bucket_values"]!), corner.BucketValues);
    }

    [Theory]
    [InlineData("pressure")]
    [InlineData("temperature")]
    public void Analysis_matches_python(string mode)
    {
        var names = ChannelNames();
        var modeNode = Fixture["modes"]![mode]!;
        var laps = modeNode["single_lap"]!.AsObject().Select(p => int.Parse(p.Key)).ToList();

        foreach (var lap in laps)
        {
            var result = TireGrip.AnalyzeTireGrip(Session.Value.FilterByLap(lap), names, mode);
            var expected = modeNode["single_lap"]![lap.ToString()]!;
            Assert.Equal(expected["metric_unit"]!.GetValue<string>(), result.MetricUnit);
            Assert.Equal(expected["accel_unit"]!.GetValue<string>(), result.AccelUnit);
            foreach (var (key, corner) in new[] { ("front_left", result.FrontLeft), ("front_right", result.FrontRight), ("rear_left", result.RearLeft), ("rear_right", result.RearRight) })
            {
                AssertCorner(expected[key]!, corner, 1e-6);
            }
        }

        var multi = TireGrip.AnalyzeTireGripMultiLap(Session.Value, laps, names, mode);
        AssertCorner(modeNode["multi_lap"]!["front_left"]!, multi.FrontLeft, 1e-6);
        AssertCorner(modeNode["multi_lap"]!["rear_right"]!, multi.RearRight, 1e-6);
    }
}

/// <summary>Unit tests for the driver and tire-grip primitives.</summary>
public class DriverTireTests
{
    [Fact]
    public void Percentile_and_linspace_match_numpy()
    {
        Assert.Equal(3.5, Numerics.Percentile(new double[] { 1, 2, 3, 4, 5, 6 }, 50));
        Assert.Equal(5.95, Numerics.Percentile(new double[] { 1, 2, 3, 4, 5, 6 }, 99), 1e-12);
        Assert.Equal(new double[] { 0, 0.25, 0.5, 0.75, 1 }, Numerics.Linspace(0, 1, 5));
    }

    [Fact]
    public void Throttle_acceptance_needs_sustained_full_throttle()
    {
        var n = 100;
        var distance = Enumerable.Range(0, n).Select(i => i * 5.0).ToArray();
        var timecodes = Enumerable.Range(0, n).Select(i => (long)i * 100).ToArray();
        var throttle = new double[n];
        var lateral = new double[n];
        for (var i = 20; i < 60; i++) lateral[i] = 1.0;
        for (var i = 40; i < 44; i++) throttle[i] = 100; // 300 ms blip: not sustained
        for (var i = 50; i < 100; i++) throttle[i] = 100;
        var corner = new Corner { Id = 1, Name = "Turn 1", StartDist = 100, EndDist = 300, ApexDist = 200 };

        var ta = DriverAnalysis.FindThrottleAcceptance(distance, timecodes, throttle, lateral, corner, throttleThreshold: 98);

        Assert.NotNull(ta);
        Assert.Equal(250, ta!.FullThrottleDist);
        Assert.True(ta.ThrottleAcceptancePct is > 0 and <= 100);
        Assert.Null(DriverAnalysis.FindThrottleAcceptance(distance, timecodes, new double[n], lateral, corner, throttleThreshold: 98));
    }
}
