using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.Tests;

/// <summary>Track store, gate crossings and split times.</summary>
public class TracksTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    [Fact]
    public void Segment_intersection_parameter_is_along_the_path()
    {
        Assert.Equal(0.25, SectorTimes.SegmentParameter(0, 0, 4, 0, 1, -1, 1, 1)!.Value, 1e-12);
        Assert.Null(SectorTimes.SegmentParameter(0, 0, 4, 0, 5, -1, 5, 1)); // chord beyond the segment
        Assert.Null(SectorTimes.SegmentParameter(0, 0, 4, 0, 1, 1, 3, 1)); // parallel
    }

    [Fact]
    public void Gate_crossings_on_a_synthetic_square_lap()
    {
        // A 400 m square lap at 10 m/s, 1 s samples: corners at t = 0, 10, 20, 30, 40.
        var origin = new GeoPoint(35.0, 139.0);
        var lat = new List<double>();
        var lon = new List<double>();
        var t = new List<long>();
        var corners = new (double X, double Y)[] { (0, 0), (100, 0), (100, 100), (0, 100), (0, 0) };
        for (var c = 0; c < 4; c++)
        {
            for (var k = 0; k < 10; k++)
            {
                var f = k / 10.0;
                var x = corners[c].X + (corners[c + 1].X - corners[c].X) * f;
                var y = corners[c].Y + (corners[c + 1].Y - corners[c].Y) * f;
                var p = GeoProjection.FromLocal(x, y, origin);
                lat.Add(p.Lat);
                lon.Add(p.Lon);
                t.Add(100_000 + (c * 10 + k) * 1000);
            }
        }

        var gate1 = new SectorGate("G1", GeoProjection.FromLocal(55, -5, origin), GeoProjection.FromLocal(55, 5, origin)); // crossed at t = 5.5 s
        var gate2 = new SectorGate("G2", GeoProjection.FromLocal(95, 50, origin), GeoProjection.FromLocal(105, 50, origin)); // crossed at t = 15 s
        var track = new TrackDefinition("square", "Square", origin, new[] { gate1, gate2 });
        var lap = new Lap(1, 100_000, 140_000, "full");

        var splits = SectorTimes.ForLap(lap, lat.ToArray(), lon.ToArray(), t.ToArray(), track);

        Assert.True(splits.Complete);
        Assert.Equal(5.5, splits.GateTimesS[0], 1e-6);
        Assert.Equal(15.0, splits.GateTimesS[1], 1e-6);
        Assert.Equal(new[] { 5.5, 9.5, 25.0 }, splits.Sectors.Select(s => Math.Round(s, 6)));

        // A gate the path never reaches leaves that sector and the rest NaN.
        var far = new SectorGate("Far", GeoProjection.FromLocal(500, 0, origin), GeoProjection.FromLocal(500, 10, origin));
        var incomplete = SectorTimes.ForLap(lap, lat.ToArray(), lon.ToArray(), t.ToArray(), track with { Gates = new[] { gate1, far } });
        Assert.False(incomplete.Complete);
        Assert.Equal(5.5, incomplete.Sectors[0], 1e-6);
        Assert.True(double.IsNaN(incomplete.Sectors[1]));
    }

    [Fact]
    public void Gates_placed_across_the_sample_lap_split_it_at_the_chosen_samples()
    {
        var log = Sample.Value;
        var profile = BuiltinOnly.GetProfileForLog(log.Metadata, log.FileName)!;
        var latName = profile.ChannelNames["gps_latitude"];
        var lonName = profile.ChannelNames["gps_longitude"];
        var lap14 = log.FilterByLap(14).SelectChannels(new[] { latName, lonName }).ResampleToChannel(latName);
        var lat = lap14[latName].Values;
        var lon = lap14[lonName].Values;
        var tc = lap14[latName].Timecodes;
        var origin = new GeoPoint(lat[0], lon[0]);
        var i1 = lat.Length / 3;
        var i2 = 2 * lat.Length / 3;
        var track = new TrackDefinition("fuji_test", "Fuji (test gates)", origin, new[]
        {
            SectorTimes.GateAcrossPath(lat, lon, i1, "G1", origin),
            SectorTimes.GateAcrossPath(lat, lon, i2, "G2", origin),
        });

        var table = SectorTimes.Build(log, new[] { 14, 12, 13 }, track, latName, lonName, TestContext.Current.CancellationToken);

        var row14 = table.Laps.First(r => r.LapNum == 14);
        Assert.True(row14.Complete);
        var lapStart = log.Laps.First(l => l.Num == 14).StartTime;
        Assert.Equal((tc[i1] - lapStart) / 1000.0, row14.GateTimesS[0], 0.05); // within one 25 Hz sample
        Assert.Equal((tc[i2] - lapStart) / 1000.0, row14.GateTimesS[1], 0.05);
        Assert.Equal(row14.LapTimeS, row14.Sectors.Sum(), 1e-9);
        Assert.All(table.Laps, r => Assert.True(r.Complete, $"lap {r.LapNum} incomplete"));
        Assert.Equal(14, table.BestLapNum);
        Assert.True(table.TheoreticalBestS <= table.Laps.Min(r => r.LapTimeS) + 1e-9);
        Assert.Equal(3, table.BestSector.Length);
    }

    [Fact]
    public void Track_store_round_trips_and_finds_nearby_tracks()
    {
        var dir = Directory.CreateTempSubdirectory("inferno-tracks");
        try
        {
            var store = new TrackStore(Path.Combine(dir.FullName, "cfg", "tracks.yaml"));
            Assert.Empty(store.LoadUser());
            var anchor = new GeoPoint(35.3717, 138.9274);
            var track = new TrackDefinition(TrackStore.IdFor("Fuji Speedway (short)"), "Fuji Speedway (short)", anchor, new[]
            {
                new SectorGate("S1", new GeoPoint(35.3720, 138.9280), new GeoPoint(35.3722, 138.9284)),
            });

            store.Save(track);
            store.Save(new TrackDefinition("elsewhere", "Elsewhere", new GeoPoint(0, 0), Array.Empty<SectorGate>()));

            Assert.Equal("fuji_speedway_short", track.Id);
            var loaded = store.LoadAll();
            Assert.Equal(2, loaded.Count);
            var fuji = loaded.First(t => t.Id == track.Id);
            Assert.Equal(track.Name, fuji.Name);
            Assert.Equal(track.Anchor, fuji.Anchor);
            Assert.Equal(track.Gates, fuji.Gates); // SectorGate and GeoPoint are value records
            Assert.Equal(track.Id, store.FindNear(new GeoPoint(35.375, 138.93))?.Id);
            Assert.Null(store.FindNear(new GeoPoint(36.0, 139.0)));

            store.Delete("elsewhere");
            Assert.Single(store.LoadAll());
            Assert.Empty(TrackYaml.Parse("not: [valid"));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
