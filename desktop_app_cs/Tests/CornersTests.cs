using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>Corner detection against Python on the Fuji 86 sample (Fixtures/corners_86.json).</summary>
public class CornersParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "corners_86.json")))!;

    public static TheoryData<string> LapNumbers()
    {
        var data = new TheoryData<string>();
        foreach (var lap in Fixture["laps"]!.AsObject())
        {
            data.Add(lap.Key);
        }

        return data;
    }

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    private static void AssertSummary(JsonNode expected, double[] values, double relative = 1e-9)
    {
        Assert.Equal(expected["length"]!.GetValue<int>(), values.Length);
        // A sum of centred coordinates cancels to ~0, so its tolerance scales
        // with the magnitudes summed rather than with the result.
        var scale = values.Length * values.Max(Math.Abs) * 1e-14;
        foreach (var (key, actual) in new[] { ("first", values[0]), ("last", values[^1]), ("min", values.Min()), ("max", values.Max()), ("sum", Numerics.Sum(values)) })
        {
            var e = expected[key]!.GetValue<double>();
            Assert.True(Math.Abs(e - actual) <= Math.Max(Math.Max(1e-9, scale), Math.Abs(e) * relative), $"{key}: expected {e}, got {actual}");
        }
    }

    [Theory]
    [MemberData(nameof(LapNumbers))]
    public void Curvature_pipeline_matches_python(string lapNum)
    {
        var lap = Fixture["laps"]![lapNum]!;
        var lat = Doubles(lap["lat"]!);
        var lon = Doubles(lap["lon"]!);

        var (x, y) = Corners.GpsToLocalXy(lat, lon);
        AssertSummary(lap["x"]!, x);
        AssertSummary(lap["y"]!, y);
        AssertSummary(lap["path_distance"]!, Corners.PathDistance(x, y));

        var curvature = Corners.ComputeCurvature(x, y);
        AssertSummary(lap["curvature"]!, curvature.Curvature, 1e-7);
        AssertSummary(lap["signed_curvature"]!, curvature.SignedCurvature, 1e-7);
        AssertSummary(lap["radius"]!, curvature.Radius, 1e-7);
    }

    [Theory]
    [MemberData(nameof(LapNumbers))]
    public void Corners_match_python(string lapNum)
    {
        var lap = Fixture["laps"]![lapNum]!;
        var corners = Corners.IdentifyCorners(Doubles(lap["lat"]!), Doubles(lap["lon"]!));

        AssertCorners(lap["corners"]!.AsArray(), corners);
    }

    /// <summary>End to end: the C# session loader feeding the C# kernel.</summary>
    [Theory]
    [MemberData(nameof(LapNumbers))]
    public void Corners_from_the_loaded_session_match_python(string lapNum)
    {
        var session = SessionLoader.Load(SessionParityTests.SamplePath(), new ProfileStore("/nonexistent/profiles.yaml"));
        var lat = Fixture["channels"]!["lat"]!.GetValue<string>();
        var lon = Fixture["channels"]!["lon"]!.GetValue<string>();

        var corners = Corners.IdentifyCorners(session.FilterByLap(int.Parse(lapNum)), lat, lon);

        AssertCorners(Fixture["laps"]![lapNum]!["corners"]!.AsArray(), corners);
    }

    private static void AssertCorners(JsonArray expected, List<Corner> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var e = expected[i]!;
            var c = actual[i];
            Assert.Equal(e["id"]!.GetValue<int>(), c.Id);
            Assert.Equal(e["name"]!.GetValue<string>(), c.Name);
            Assert.Equal(e["direction"]!.GetValue<string>(), c.Direction.ToString());
            Assert.Equal(e["start_idx"]!.GetValue<int>(), c.StartIdx);
            Assert.Equal(e["end_idx"]!.GetValue<int>(), c.EndIdx);
            Assert.Equal(e["apex_idx"]!.GetValue<int>(), c.ApexIdx);
            Assert.Equal(e["start_dist"]!.GetValue<double>(), c.StartDist, 1e-6);
            Assert.Equal(e["end_dist"]!.GetValue<double>(), c.EndDist, 1e-6);
            Assert.Equal(e["apex_dist"]!.GetValue<double>(), c.ApexDist, 1e-6);
            Assert.Equal(e["max_curvature"]!.GetValue<double>(), c.MaxCurvature, 1e-9);
        }
    }
}

/// <summary>Synthetic cases mirroring tests/test_corners.py.</summary>
public class CornersTests
{
    [Fact]
    public void Straight_line_has_zero_curvature()
    {
        var x = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        var y = new double[100];

        var result = Corners.ComputeCurvature(x, y);

        Assert.All(result.Curvature, c => Assert.True(c < 1e-10));
        Assert.All(result.Radius, r => Assert.Equal(10000, r));
    }

    [Fact]
    public void Circle_has_constant_curvature_with_left_positive()
    {
        const double radius = 50;
        var n = 400;
        var x = new double[n];
        var y = new double[n];
        for (var i = 0; i < n; i++)
        {
            var theta = 2 * Math.PI * i / n;
            x[i] = radius * Math.Cos(theta);
            y[i] = radius * Math.Sin(theta); // counter-clockwise = turning left
        }

        var result = Corners.ComputeCurvature(x, y, posSmoothWindow: 1, curvSmoothWindow: 1);

        var middle = result.Curvature[20..^20];
        Assert.All(middle, c => Assert.Equal(1.0 / radius, c, 1e-3));
        Assert.All(result.SignedCurvature[20..^20], c => Assert.True(c > 0));
    }

    [Fact]
    public void Rolling_mean_matches_pandas_centering()
    {
        double[] v = { 1, 2, 3, 4, 5, 6 };

        // window 3: symmetric [i-1, i+1]; ends shrink (min_periods=1)
        Assert.Equal(new[] { 1.5, 2, 3, 4, 5, 5.5 }, Numerics.RollingMeanCentered(v, 3));
        // window 4: offset 1 → [i-2, i+1]
        Assert.Equal(new[] { 1.5, 2, 2.5, 3.5, 4.5, 5 }, Numerics.RollingMeanCentered(v, 4));
    }

    [Theory]
    [InlineData(1, 100000000.0)]
    [InlineData(7, 700000002.625)]
    [InlineData(8, 800000002.625)]
    [InlineData(9, 900000002.75)]
    [InlineData(127, 12700000047.250004)]
    [InlineData(128, 12800000047.375004)]
    [InlineData(129, 12900000047.625006)]
    [InlineData(300, 30000000112.12504)]
    [InlineData(1000, 100000000374.6255)]
    [InlineData(3126, 312600001171.5049)]
    public void Pairwise_sum_matches_numpy_bit_for_bit(int n, double expected)
    {
        // np.sum(1e8 + (i % 7) * 0.125 + i * 1e-9) for i in range(n), computed by numpy.
        var values = Enumerable.Range(0, n).Select(i => 1e8 + (i % 7) * 0.125 + i * 1e-9).ToArray();

        Assert.Equal(expected, Numerics.Sum(values));
    }

    [Fact]
    public void Gradient_matches_numpy()
    {
        Assert.Equal(new double[] { 1, 2, 2, 1 }, Numerics.Gradient(new double[] { 0, 1, 4, 5 }));
    }

    [Fact]
    public void Same_direction_corners_within_min_gap_are_merged_and_numbered()
    {
        var n = 300;
        var distance = Enumerable.Range(0, n).Select(i => i * 1.0).ToArray();
        var signed = new double[n];
        // Two left corners 30 m apart, then a right corner.
        for (var i = 50; i < 80; i++) signed[i] = 0.02;
        for (var i = 110; i < 140; i++) signed[i] = 0.03;
        for (var i = 250; i < 290; i++) signed[i] = -0.02;
        var curvature = signed.Select(Math.Abs).ToArray();

        var corners = Corners.IdentifyCornersFromCurvature(distance, curvature, signed, minGap: 80);

        Assert.Equal(2, corners.Count);
        Assert.Equal(("Turn 1", 'L', 50, 139, 0.03), (corners[0].Name, corners[0].Direction, corners[0].StartIdx, corners[0].EndIdx, corners[0].MaxCurvature));
        Assert.Equal(("Turn 2", 'R', 250, 289), (corners[1].Name, corners[1].Direction, corners[1].StartIdx, corners[1].EndIdx));

        var separate = Corners.IdentifyCornersFromCurvature(distance, curvature, signed, minGap: 10);
        Assert.Equal(3, separate.Count);
    }

    [Fact]
    public void Short_corners_are_dropped_and_direction_flips_split()
    {
        var n = 200;
        var distance = Enumerable.Range(0, n).Select(i => i * 1.0).ToArray();
        var signed = new double[n];
        for (var i = 50; i < 60; i++) signed[i] = 0.02; // 9 m: too short
        for (var i = 100; i < 130; i++) signed[i] = 0.02;
        for (var i = 130; i < 160; i++) signed[i] = -0.02; // flips without leaving the threshold
        var curvature = signed.Select(Math.Abs).ToArray();

        var corners = Corners.IdentifyCornersFromCurvature(distance, curvature, signed);

        Assert.Equal(new[] { 'L', 'R' }, corners.Select(c => c.Direction));
        Assert.Equal(100, corners[0].StartIdx);
        Assert.Equal(129, corners[0].EndIdx);
        Assert.Equal(130, corners[1].StartIdx);
        Assert.Equal(50, corners[0].Radius, 1e-9);
    }
}
