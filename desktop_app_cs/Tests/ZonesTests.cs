using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>Zone detection, segments and stats against Python on the Fuji 86 sample (Fixtures/zones_86.json).</summary>
public class ZonesParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "zones_86.json")))!;

    private static double[] Doubles(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();

    private static List<Zone> ZonesOf(JsonNode node) =>
        node.AsArray().Select(z => new Zone(z![0]!.GetValue<double>(), z[1]!.GetValue<double>())).ToList();

    private static List<int> LapNums => Fixture["laps"]!.AsObject().Select(p => int.Parse(p.Key)).ToList();

    private static Zones.LapArrays LapArrays(int num)
    {
        var lap = Fixture["laps"]![num.ToString()]!;
        var inputs = lap["inputs"]!;
        return new Zones.LapArrays(num, lap["lap_time_s"]!.GetValue<double>(), Doubles(inputs["distance"]!), Doubles(inputs["brake"]!), Doubles(inputs["throttle"]!), Doubles(inputs["speed_ms"]!), Doubles(inputs["speed_kmh"]!));
    }

    private static List<Corner> CornersFromFixture() =>
        Fixture["corners"]!.AsArray().Select(c => new Corner
        {
            Id = c!["id"]!.GetValue<int>(),
            Name = c["name"]!.GetValue<string>(),
            Direction = c["direction"]!.GetValue<string>()[0],
            StartIdx = c["start_idx"]!.GetValue<int>(),
            EndIdx = c["end_idx"]!.GetValue<int>(),
            StartDist = c["start_dist"]!.GetValue<double>(),
            EndDist = c["end_dist"]!.GetValue<double>(),
            ApexIdx = c["apex_idx"]!.GetValue<int>(),
            ApexDist = c["apex_dist"]!.GetValue<double>(),
            MaxCurvature = c["max_curvature"]!.GetValue<double>(),
        }).ToList();

    private static void AssertZones(JsonNode expected, List<Zone> actual)
    {
        var e = ZonesOf(expected);
        Assert.Equal(e.Count, actual.Count);
        for (var i = 0; i < e.Count; i++)
        {
            Assert.Equal(e[i].Start, actual[i].Start, 1e-9);
            Assert.Equal(e[i].End, actual[i].End, 1e-9);
        }
    }

    [Fact]
    public void Single_lap_zones_match_python()
    {
        foreach (var num in LapNums)
        {
            var lap = LapArrays(num);
            var (braking, accel) = Zones.IdentifyZonesSingleLap(lap.Distance, lap.Brake, lap.Throttle, lap.SpeedMs);

            AssertZones(Fixture["laps"]![num.ToString()]!["braking_zones"]!, braking);
            AssertZones(Fixture["laps"]![num.ToString()]!["accel_zones"]!, accel);
        }
    }

    [Fact]
    public void Averaged_zones_match_python()
    {
        var laps = LapNums.Select(LapArrays).ToList();
        var (braking, accel) = Zones.DetectZonesFromArrays(laps);

        AssertZones(Fixture["braking_zones"]!, braking);
        AssertZones(Fixture["accel_zones"]!, accel);
    }

    [Fact]
    public void Segments_match_python()
    {
        var laps = LapNums.Select(LapArrays).ToList();
        var (braking, accel) = Zones.DetectZonesFromArrays(laps);
        var segments = Zones.CreateTrackSegments(CornersFromFixture(), braking, accel, Fixture["track_length"]!.GetValue<double>());

        var expected = Fixture["segments"]!.AsArray();
        Assert.Equal(expected.Count, segments.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var e = expected[i]!;
            var s = segments[i];
            Assert.Equal(e["id"]!.GetValue<int>(), s.Id);
            Assert.Equal(e["segment_type"]!.GetValue<string>(), s.SegmentType);
            Assert.Equal(e["name"]!.GetValue<string>(), s.Name);
            Assert.Equal(e["corner_id"]!.GetValue<int>(), s.CornerId);
            Assert.Equal(e["start_dist"]!.GetValue<double>(), s.StartDist, 1e-9);
            Assert.Equal(e["end_dist"]!.GetValue<double>(), s.EndDist, 1e-9);
            Assert.Equal(e["apex_dist"]?.GetValue<double>(), s.ApexDist);
        }
    }

    private static List<TrackSegment> SegmentsFromFixture() =>
        Fixture["segments"]!.AsArray().Select(e => new TrackSegment(
            e!["id"]!.GetValue<int>(), e["segment_type"]!.GetValue<string>(), e["start_dist"]!.GetValue<double>(),
            e["end_dist"]!.GetValue<double>(), e["name"]!.GetValue<string>(), e["corner_id"]!.GetValue<int>(), e["apex_dist"]?.GetValue<double>())).ToList();

    private static void AssertNullable(JsonNode row, string key, double? actual, double tolerance = 1e-9)
    {
        var expected = row[key]?.GetValue<double>();
        if (expected is null)
        {
            Assert.True(actual is null, $"{key}: expected null, got {actual}");
        }
        else
        {
            Assert.True(actual is not null, $"{key}: expected {expected}, got null");
            Assert.Equal(expected.Value, actual!.Value, Math.Max(tolerance, Math.Abs(expected.Value) * 1e-12));
        }
    }

    [Fact]
    public void Segment_stats_match_python()
    {
        var laps = LapNums.Select(LapArrays).ToList();
        var stats = Zones.ComputeSegmentStats(laps, SegmentsFromFixture());

        var expected = Fixture["segment_stats"]!.AsArray();
        Assert.Equal(expected.Count, stats.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var e = expected[i]!;
            var s = stats[i];
            Assert.Equal(e["segment_id"]!.GetValue<int>(), s.SegmentId);
            Assert.Equal(e["segment_type"]!.GetValue<string>(), s.SegmentType);
            Assert.Equal(e["lap_num"]!.GetValue<int>(), s.LapNum);
            Assert.Equal(e["lap_time"]!.GetValue<double>(), s.LapTime, 1e-9);
            AssertNullable(e, "braking_point", s.BrakingPoint);
            AssertNullable(e, "brake_offset", s.BrakeOffset);
            AssertNullable(e, "peak_brake", s.PeakBrake);
            AssertNullable(e, "peak_brake_dist", s.PeakBrakeDist);
            AssertNullable(e, "brake_release_point", s.BrakeReleasePoint);
            AssertNullable(e, "brake_release_offset", s.BrakeReleaseOffset);
            AssertNullable(e, "entry_speed", s.EntrySpeed);
            AssertNullable(e, "braking_distance", s.BrakingDistance);
            AssertNullable(e, "mean_decel_g", s.MeanDecelG);
            AssertNullable(e, "min_speed", s.MinSpeed);
            AssertNullable(e, "exit_speed", s.ExitSpeed);
            AssertNullable(e, "throttle_point", s.ThrottlePoint);
            AssertNullable(e, "throttle_offset", s.ThrottleOffset);
        }
    }

    [Theory]
    [InlineData("g_utilization", true)]
    [InlineData("g_utilization_derived_inline", false)]
    public void G_utilization_matches_python(string key, bool useInline)
    {
        var hasInline = Fixture["channels"]!["inline_g"] is not null;
        if (useInline && !hasInline)
        {
            return;
        }

        var laps = LapNums.Select(num =>
        {
            var inputs = Fixture["laps"]![num.ToString()]!["inputs"]!;
            return new Zones.LapGArrays(num, Doubles(inputs["distance"]!), Doubles(inputs["speed_kmh"]!), Doubles(inputs["lateral_g"]!), useInline ? Doubles(inputs["inline_g"]!) : null);
        }).ToList();

        var rows = Zones.ComputeGUtilization(laps, SegmentsFromFixture(), CornersFromFixture());

        var expected = Fixture[key]!.AsArray();
        Assert.Equal(expected.Count, rows.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            var e = expected[i]!;
            var r = rows[i];
            Assert.Equal(e["corner_id"]!.GetValue<int>(), r.CornerId);
            Assert.Equal(e["lap_num"]!.GetValue<int>(), r.LapNum);
            Assert.Equal(e["total_g_min_phase"]!.GetValue<string>(), r.TotalGMinPhase);
            AssertNullable(e, "total_g_mean", r.TotalGMean, 1e-9);
            AssertNullable(e, "total_g_max", r.TotalGMax);
            AssertNullable(e, "total_g_min", r.TotalGMin);
            AssertNullable(e, "total_g_min_dist", r.TotalGMinDist);
            AssertNullable(e, "g_utilization_pct", r.GUtilizationPct, 1e-9);
            AssertNullable(e, "early_braking_coast_m", r.EarlyBrakingCoastM);
            AssertNullable(e, "braking_g_mean", r.BrakingGMean, 1e-9);
            AssertNullable(e, "entry_g_mean", r.EntryGMean, 1e-9);
            AssertNullable(e, "mid_g_mean", r.MidGMean, 1e-9);
            AssertNullable(e, "exit_g_mean", r.ExitGMean, 1e-9);
        }
    }

    /// <summary>End to end from the C# loader: the lap arrays must reproduce the fixture inputs.</summary>
    [Fact]
    public void Lap_arrays_from_the_loaded_session_match_python()
    {
        var session = SessionLoader.Load(SessionParityTests.SamplePath(), new ProfileStore("/nonexistent/profiles.yaml"));
        var ch = Fixture["channels"]!;
        foreach (var num in LapNums)
        {
            var expected = LapArrays(num);
            var actual = Zones.ExtractLap(session, num, ch["throttle"]!.GetValue<string>(), ch["brake"]!.GetValue<string>(), ch["speed"]!.GetValue<string>());

            Assert.Equal(expected.LapTimeS, actual.LapTimeS, 1e-9);
            foreach (var (name, e, a) in new[] { ("distance", expected.Distance, actual.Distance), ("brake", expected.Brake, actual.Brake), ("throttle", expected.Throttle, actual.Throttle), ("speed_ms", expected.SpeedMs, actual.SpeedMs), ("speed_kmh", expected.SpeedKmh, actual.SpeedKmh) })
            {
                Assert.Equal(e.Length, a.Length);
                for (var i = 0; i < e.Length; i++)
                {
                    Assert.True(Math.Abs(e[i] - a[i]) <= Math.Max(1e-9, Math.Abs(e[i]) * 1e-9), $"lap {num} {name}[{i}]: {e[i]} vs {a[i]}");
                }
            }
        }
    }
}

/// <summary>Synthetic cases for zone logic and lap selection.</summary>
public class ZonesTests
{
    [Fact]
    public void Zones_split_on_threshold_and_accel_merges_across_short_gaps()
    {
        var n = 100;
        var distance = Enumerable.Range(0, n).Select(i => i * 10.0).ToArray(); // 10 m per sample
        var speed = Enumerable.Repeat(50.0, n).ToArray(); // m/s → 0.2 s per sample
        var brake = new double[n];
        var throttle = new double[n];
        for (var i = 10; i < 20; i++) brake[i] = 60;
        for (var i = 20; i < 40; i++) throttle[i] = 100;
        for (var i = 43; i < 60; i++) throttle[i] = 100; // 30 m gap = 0.6 s: merged
        for (var i = 70; i < 80; i++) throttle[i] = 100; // 100 m gap = 2 s: separate

        var (braking, accel) = Zones.IdentifyZonesSingleLap(distance, brake, throttle, speed);

        Assert.Equal(new[] { new Zone(100, 190) }, braking);
        Assert.Equal(new[] { new Zone(200, 590), new Zone(700, 790) }, accel);
    }

    [Fact]
    public void Averaging_votes_per_metre()
    {
        var lapA = new List<Zone> { new(10, 20) };
        var lapB = new List<Zone> { new(15, 30) };
        var lapC = new List<Zone> { new(25, 40) };
        var none = new List<Zone>();

        // Two laps at threshold 0.5 need one vote: the union.
        var (two, _) = Zones.AverageZonesAcrossLaps(new[] { lapA, lapB }, new[] { none, none }, trackLength: 50);
        Assert.Equal(new[] { new Zone(10, 30) }, two);

        // Three laps need 1.5 votes, i.e. two laps agreeing.
        var (three, _) = Zones.AverageZonesAcrossLaps(new[] { lapA, lapB, lapC }, new[] { none, none, none }, trackLength: 50);
        Assert.Equal(new[] { new Zone(15, 20), new Zone(25, 30) }, three);
    }

    [Fact]
    public void Segments_default_to_100m_braking_and_10m_minimum_exit()
    {
        var corner = new Corner { Id = 1, Name = "Turn 1", Direction = 'L', StartDist = 500, EndDist = 600, ApexDist = 550, MaxCurvature = 0.01 };

        var segments = Zones.CreateTrackSegments(new[] { corner }, new List<Zone>(), new List<Zone>(), 1000);

        Assert.Equal(new[] { ("braking", 400.0, 500.0), ("corner", 500.0, 600.0), ("acceleration", 600.0, 650.0) }, segments.Select(s => (s.SegmentType, s.StartDist, s.EndDist)));
        Assert.Equal(550, segments[1].ApexDist);
    }

    [Fact]
    public void Top_laps_use_full_laps_within_103_percent()
    {
        var laps = new List<Lap>
        {
            new(0, 0, 200_000, "out"),
            new(1, 200_000, 300_000, "full"),
            new(2, 300_000, 402_000, "full"),
            new(3, 402_000, 506_000, "full"),
            new(4, 506_000, 700_000, "in"),
        };

        Assert.Equal(new[] { 1, 2 }, LapSelection.GetTopLaps(laps).Select(l => l.Num));
        Assert.Equal(1, LapSelection.GetBestLap(laps).Num);

        var untyped = laps.Select(l => l with { LapType = "" }).ToList();
        Assert.Equal(new[] { 1, 2 }, LapSelection.GetTopLaps(untyped).Select(l => l.Num)); // first and last excluded
    }
}
