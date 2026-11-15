using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.Tests;

/// <summary>Track regions (corners and sectors as gate pairs) and the analyses cut down to them.</summary>
public class TrackRegionTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static IReadOnlyDictionary<string, string> Names()
    {
        var names = new Dictionary<string, string>(ChannelKeys.DefaultChannelNames);
        foreach (var (key, value) in BuiltinOnly.GetProfileForLog(Sample.Value.Metadata, Path.GetFileName(SessionParityTests.SamplePath()))!.ChannelNames)
        {
            names[key] = value;
        }

        return names;
    }

    private static readonly Lazy<DriverConsistencyResult> Corners = new(() =>
        DriverConsistency.Analyze(Sample.Value, LapSelection.GetTopLaps(Sample.Value.Laps).Select(l => l.Num).ToList(), Names()));

    [Fact]
    public void Each_detected_corner_becomes_a_region_from_its_braking_to_its_acceleration()
    {
        var result = Corners.Value;
        var regions = TrackRegions.FromCorners(result);

        Assert.Equal(result.Corners.Count, regions.Count);
        Assert.Equal(result.Corners.Select(c => c.Name), regions.Select(r => r.Name));
        Assert.Equal(Enumerable.Range(0, regions.Count).Select(i => $"corner:{i}"), regions.Select(r => r.Id));
        Assert.All(regions, r => Assert.Equal(TrackRegionKind.Corner, r.Kind));
        Assert.All(regions.Skip(1).SkipLast(1), r => Assert.True(r.Start is not null && r.End is not null));
    }

    [Fact]
    public void A_corner_region_cuts_every_lap_where_the_car_drove_through_it()
    {
        var log = Sample.Value;
        var names = Names();
        var regions = TrackRegions.FromCorners(Corners.Value);
        var laps = LapSelection.GetTopLaps(log.Laps).Select(l => l.Num).ToList();
        var third = regions[2];

        var windows = TrackRegions.WindowsByLap(log, laps, new[] { third }, names["gps_latitude"], names["gps_longitude"]);

        Assert.Equal(laps.Count, windows.Count);
        var durations = new List<double>();
        foreach (var num in laps)
        {
            var lap = log.Laps.First(l => l.Num == num);
            var window = Assert.Single(windows[num]); // the car went through the corner once
            Assert.InRange(window.Start, lap.StartTime, lap.EndTime);
            Assert.InRange(window.End, window.Start + 1, lap.EndTime);
            durations.Add((window.End - window.Start) / 1000.0);
        }

        // The same stretch of road every lap: a few seconds, and about the same on every lap.
        Assert.InRange(durations.Min(), 2, 40);
        Assert.True(durations.Max() - durations.Min() < 3, string.Join(", ", durations));
    }

    [Fact]
    public void Sectors_of_a_track_cover_the_lap_between_its_gates()
    {
        var log = Sample.Value;
        var names = Names();
        var lapNum = LapSelection.GetBestLap(log.Laps)!.Num;
        var lap = log.Laps.First(l => l.Num == lapNum);
        var gps = log.FilterByLap(lapNum, new[] { names["gps_latitude"], names["gps_longitude"] }).ResampleToChannel(names["gps_latitude"]);
        var lat = gps[names["gps_latitude"]].Values;
        var lon = gps[names["gps_longitude"]].Values;
        var anchor = new GeoPoint(lat[0], lon[0]);
        var track = new TrackDefinition("t", "T", anchor, new[]
        {
            SectorTimes.GateAcrossPath(lat, lon, lat.Length / 3, "G1", anchor),
            SectorTimes.GateAcrossPath(lat, lon, 2 * lat.Length / 3, "G2", anchor),
        });

        var sectors = TrackRegions.FromSectors(track);
        Assert.Equal(new[] { "S1", "S2", "S3" }, sectors.Select(s => s.Name));
        Assert.Null(sectors[0].Start);
        Assert.Null(sectors[^1].End);
        Assert.Empty(TrackRegions.FromSectors(track with { Gates = Array.Empty<SectorGate>() }));

        var timecodes = gps[names["gps_latitude"]].Timecodes;
        var each = sectors.Select(s => TrackRegions.Window(s, lap, lat, lon, timecodes)!.Value).ToList();
        Assert.Equal(lap.StartTime, each[0].Start);
        Assert.Equal(each[0].End, each[1].Start);
        Assert.Equal(each[1].End, each[2].Start);
        Assert.Equal(lap.EndTime, each[2].End);

        // Ticking all three is the whole lap; ticking two that touch is one stretch.
        Assert.Equal(new[] { (lap.StartTime, lap.EndTime) }, TrackRegions.Windows(sectors, lap, lat, lon, timecodes));
        Assert.Single(TrackRegions.Windows(sectors.Take(2), lap, lat, lon, timecodes));
    }

    [Fact]
    public void Windows_merge_and_contain()
    {
        var merged = TrackRegions.Merge(new[] { (50L, 60L), (10L, 20L), (15L, 30L), (30L, 35L) });
        Assert.Equal(new[] { (10L, 35L), (50L, 60L) }, merged);
        Assert.True(TrackRegions.Contains(merged, 10));
        Assert.True(TrackRegions.Contains(merged, 35));
        Assert.False(TrackRegions.Contains(merged, 36));
        Assert.True(TrackRegions.Contains(merged, 55));
        Assert.False(TrackRegions.Contains(merged, 61));
        Assert.Equal(0.035, TrackRegions.DurationS(merged), 9);
    }

    [Fact]
    public void Suspension_velocities_of_a_region_are_the_whole_laps_cut_down()
    {
        var log = Sample.Value;
        var names = Names();
        var laps = LapSelection.GetTopLaps(log.Laps).Select(l => l.Num).ToList();
        var regions = TrackRegions.FromCorners(Corners.Value);
        var whole = Suspension.AnalyzeSuspensionVelocityMultiLap(log, laps, names, cancellation: TestContext.Current.CancellationToken);

        var windows = TrackRegions.WindowsByLap(log, laps, new[] { regions[2] }, names["gps_latitude"], names["gps_longitude"]);
        var corner = Suspension.AnalyzeSuspensionVelocityMultiLap(log, laps, names, cancellation: TestContext.Current.CancellationToken, windowsByLap: windows);

        // The same samples as the whole laps hold, only fewer: about the share of the lap the corner takes.
        var share = (double)corner.FrontLeft.Velocity.Length / whole.FrontLeft.Velocity.Length;
        var timeShare = windows.Values.Sum(TrackRegions.DurationS) / laps.Sum(n => log.Laps.First(l => l.Num == n).LapTimeMs / 1000.0);
        Assert.InRange(share, 0.01, 0.5);
        Assert.Equal(timeShare, share, 2);
        Assert.Equal(corner.FrontLeft.Velocity.Length, corner.RearRight.Velocity.Length);

        // Every region ticked that tiles the lap gives the whole lap back.
        var all = laps.ToDictionary(n => n, n => (IReadOnlyList<(long Start, long End)>)new[] { (log.Laps.First(l => l.Num == n).StartTime, log.Laps.First(l => l.Num == n).EndTime) });
        var same = Suspension.AnalyzeSuspensionVelocityMultiLap(log, laps, names, cancellation: TestContext.Current.CancellationToken, windowsByLap: all);
        Assert.Equal(whole.FrontLeft.Velocity, same.FrontLeft.Velocity);

        // A lap with no windows contributes nothing.
        var none = Suspension.AnalyzeSuspensionVelocityMultiLap(
            log, laps, names, cancellation: TestContext.Current.CancellationToken, windowsByLap: new Dictionary<int, IReadOnlyList<(long Start, long End)>>());
        Assert.Empty(none.FrontLeft.Velocity);
    }
}
