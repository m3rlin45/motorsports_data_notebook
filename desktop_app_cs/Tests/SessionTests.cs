using System.Text.Json.Nodes;
using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// The data layer against the Python stack on the Fuji 86 sample:
/// <c>Fixtures/session_86.json</c> is what <c>libxrk.aim_xrk</c> (Rust backend)
/// and <c>load_session</c> produce (<c>tools/extract_session_fixture.py</c>);
/// <c>App/fixtures/laps.json</c> is the Python lap pipeline's output
/// (<c>tools/extract_fixture.py</c>).
/// </summary>
public class SessionParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "session_86.json")))!;

    /// <summary>The repo's sample session; found by walking up from the test directory.</summary>
    public static string SamplePath()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "pyproject.toml")))
        {
            dir = dir.Parent;
        }

        Assert.NotNull(dir);
        var path = Path.Combine(dir!.FullName, "workspace_template", "data", Fixture["sample"]!.GetValue<string>());
        Assert.True(File.Exists(path), $"sample missing: {path}");
        return path;
    }

    private static readonly Lazy<LogFile> Raw = new(() => XrkReader.Read(SamplePath()));

    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");

    [Fact]
    public void Native_library_loads()
    {
        Assert.Matches(@"^\d+\.\d+\.\d+", XrkReader.NativeVersion);
    }

    /// <summary>
    /// Python's channel order (the installed libxrk 0.13.0) comes from a Rust
    /// HashMap and is not deterministic; the FFI (libxrk master) sorts by
    /// channel index. Compare as sets.
    /// </summary>
    [Fact]
    public void Channel_set_matches_python()
    {
        var expected = Fixture["channel_order"]!.AsArray().Select(n => n!.GetValue<string>()).OrderBy(n => n, StringComparer.Ordinal);
        Assert.Equal(expected, Raw.Value.ChannelNames.OrderBy(n => n, StringComparer.Ordinal));
    }

    public static TheoryData<string> ChannelNames()
    {
        var data = new TheoryData<string>();
        foreach (var name in Fixture["channel_order"]!.AsArray())
        {
            data.Add(name!.GetValue<string>());
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(ChannelNames))]
    public void Channel_matches_python(string name)
    {
        var expected = Fixture["channels"]![name]!;
        var channel = Raw.Value[name];

        Assert.Equal(expected["units"]!.GetValue<string>(), channel.Units);
        Assert.Equal(expected["dec_pts"]!.GetValue<int>(), channel.DecPts);
        Assert.Equal(expected["interpolate"]!.GetValue<bool>(), channel.Interpolate);
        AssertSummary(expected["timecodes"]!, channel.Timecodes.Select(t => (double)t).ToArray());
        AssertSummary(expected["values"]!, channel.Values);
    }

    private static void AssertSummary(JsonNode expected, double[] values)
    {
        Assert.Equal(expected["length"]!.GetValue<int>(), values.Length);
        if (values.Length == 0)
        {
            return;
        }

        // Relative tolerance: libxrk master reorders the GPS float arithmetic
        // (commit fe0f9074) relative to the 0.13.0 wheel Python runs, which
        // moves the last bits of the derived GPS channels.
        AssertClose(expected["first"]!.GetValue<double>(), values[0]);
        AssertClose(expected["last"]!.GetValue<double>(), values[^1]);
        AssertClose(expected["min"]!.GetValue<double>(), values.Min());
        AssertClose(expected["max"]!.GetValue<double>(), values.Max());
        AssertClose(expected["sum"]!.GetValue<double>(), values.Sum());
    }

    private static void AssertClose(double expected, double actual, double relative = 1e-9)
    {
        Assert.Equal(expected, actual, Math.Max(1e-9, Math.Abs(expected) * relative));
    }

    [Fact]
    public void Raw_laps_match_python()
    {
        var expected = Fixture["laps_raw"]!.AsArray()
            .Select(l => new Lap(l!["num"]!.GetValue<int>(), l["start_time"]!.GetValue<long>(), l["end_time"]!.GetValue<long>(), l["lap_type"]!.GetValue<string>(), l["session"]?.GetValue<int>()))
            .ToList();
        Assert.Equal(expected, Raw.Value.Laps);
    }

    [Fact]
    public void Scalar_metadata_matches_python()
    {
        foreach (var (key, value) in Fixture["metadata"]!.AsObject())
        {
            Assert.True(Raw.Value.Metadata.TryGetValue(key, out var actual), $"missing metadata key {key}");
            Assert.Equal(value!.GetValue<string>(), actual);
        }

        Assert.Equal("Toyota 86 ZN6", BuiltinOnly.GetProfileForLog(Raw.Value.Metadata, Raw.Value.FileName)?.Name);
    }

    [Fact]
    public void Load_session_derived_channels_and_laps_match_python()
    {
        var session = SessionLoader.PostProcess(Raw.Value, BuiltinOnly, SessionLoader.DefaultLapTypes);
        var expected = Fixture["session"]!;

        var laps = expected["laps"]!.AsArray().Select(l => (l!["num"]!.GetValue<int>(), l["lap_time_s"]!.GetValue<double>())).ToList();
        Assert.Equal(laps, session.Laps.Select(l => (l.Num, l.LapTimeS)).ToList());
        Assert.All(session.Laps, l => Assert.Equal("full", l.LapType));

        AssertSummary(expected["speed_kmh"]!, session["speed_kmh"].Values);
        AssertSummary(expected["distance_m"]!, session["distance_m"].Values);

        var distance = session["distance_m"];
        foreach (var (num, value) in expected["lap_end_distance_m"]!.AsObject())
        {
            var lap = session.Laps.Single(l => l.Num == int.Parse(num));
            var end = Channel.LowerBound(distance.Timecodes, lap.EndTime);
            Assert.Equal(value!.GetValue<double>(), distance.Values[end - 1], 1e-6);
        }
    }

    /// <summary>
    /// The Python lap pipeline used by the spike fixture, replayed in C#:
    /// filter_by_lap → select_channels → resample_to_channel(GPS Speed).
    /// laps.json holds values rounded to 3 decimals (time 4, distance 2).
    /// </summary>
    [Fact]
    public void Lap_pipeline_matches_the_spike_fixture()
    {
        var fixture = InfernoAnalyzer.App.Fixture.Load(Path.Combine(AppContext.BaseDirectory, "fixtures", "laps.json"));
        var session = SessionLoader.PostProcess(Raw.Value, BuiltinOnly, SessionLoader.DefaultLapTypes);
        var profile = BuiltinOnly.GetProfileForLog(session.Metadata, session.FileName)!;
        var names = new Dictionary<string, string>
        {
            ["Speed"] = "speed_kmh",
            ["Throttle"] = profile.ChannelNames["throttle"],
            ["Brake"] = profile.ChannelNames["brake"],
        };

        foreach (var lapFixture in fixture.Laps)
        {
            var lap = session.FilterByLap(lapFixture.Num)
                .SelectChannels(names.Values.Append("distance_m").Append(fixture.Timebase).Distinct())
                .ResampleToChannel(fixture.Timebase);

            var t = lap[fixture.Timebase].Timecodes;
            Assert.Equal(lapFixture.TimeS.Length, t.Length);
            for (var i = 0; i < t.Length; i++)
            {
                Assert.Equal(lapFixture.TimeS[i], (t[i] - t[0]) / 1000.0, 5e-5);
            }

            AssertRounded($"lap {lapFixture.Num} distance_m", lapFixture.DistanceM, lap["distance_m"].Values, 0.005);
            foreach (var (display, channel) in names)
            {
                AssertRounded($"lap {lapFixture.Num} {display}", lapFixture.Series[display], lap[channel].Values, 0.0005);
            }
        }
    }

    private static void AssertRounded(string label, double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance + 1e-9, $"{label} index {i} of {expected.Length}: expected {expected[i]}, got {actual[i]}");
        }
    }
}

/// <summary>LogFile operations on synthetic data, mirroring libxrk's Python tests.</summary>
public class LogFileTests
{
    private static LogFile Session()
    {
        var interp = new Channel("Interp", new long[] { 0, 10, 20, 30 }, new double[] { 0, 10, 20, 30 }, new Dictionary<string, string> { ["interpolate"] = "True", ["units"] = "x" });
        var step = new Channel("Step", new long[] { 5, 15, 25 }, new double[] { 1, 2, 3 }, new Dictionary<string, string> { ["interpolate"] = "False" });
        var laps = new List<Lap> { new(0, 0, 12, "out"), new(1, 12, 24, "full"), new(2, 24, 40, "full") };
        return new LogFile(new[] { interp, step }, laps, new Dictionary<string, string>(), "synthetic");
    }

    [Fact]
    public void Filter_by_time_range_is_half_open_and_keeps_overlapping_laps()
    {
        var filtered = Session().FilterByTimeRange(10, 25);

        Assert.Equal(new long[] { 10, 20 }, filtered["Interp"].Timecodes);
        Assert.Equal(new long[] { 15 }, filtered["Step"].Timecodes);
        Assert.Equal(new[] { 0, 1, 2 }, filtered.Laps.Select(l => l.Num));
    }

    [Fact]
    public void Filter_by_lap_uses_the_first_matching_row()
    {
        var lap = Session().FilterByLap(1);

        Assert.Equal(new long[] { 20 }, lap["Interp"].Timecodes);
        Assert.Equal(new long[] { 15 }, lap["Step"].Timecodes);
        Assert.Throws<ArgumentException>(() => Session().FilterByLap(9));
    }

    [Fact]
    public void Resample_interpolates_or_forward_fills_per_metadata()
    {
        var target = new long[] { -5, 0, 5, 12, 27, 35 };
        var resampled = Session().ResampleToTimecodes(target);

        Assert.Equal(new double[] { 0, 0, 5, 12, 27, 30 }, resampled["Interp"].Values);
        Assert.Equal(new double[] { 1, 1, 1, 1, 3, 3 }, resampled["Step"].Values);
        Assert.Equal(target, resampled["Step"].Timecodes);
        Assert.Equal("x", resampled["Interp"].Units);
    }

    [Fact]
    public void Select_channels_rejects_unknown_names()
    {
        var ex = Assert.Throws<KeyNotFoundException>(() => Session().SelectChannels(new[] { "Interp", "Nope" }));
        Assert.Contains("Nope", ex.Message);
        Assert.Equal(new[] { "Step", "Interp" }, Session().SelectChannels(new[] { "Step", "Interp" }).ChannelNames);
    }

    [Fact]
    public void Clean_laps_drops_lap_zero_and_keeps_the_longest_duplicate()
    {
        var laps = new List<Lap> { new(0, 0, 10, "out"), new(1, 10, 20, "full"), new(1, 10, 25, "full"), new(2, 25, 30, "full") };

        var cleaned = SessionLoader.CleanLaps(laps);

        Assert.Equal(new[] { (1, 25L), (2, 30L) }, cleaned.Select(l => (l.Num, l.EndTime)));
        Assert.Equal(3, SessionLoader.CleanLaps(laps, dropLapZero: false).Count);
        var onlyLapZero = new List<Lap> { new(0, 0, 10, "out") };
        Assert.Same(onlyLapZero, SessionLoader.CleanLaps(onlyLapZero));
    }

    [Fact]
    public void Distance_integrates_speed_within_each_lap()
    {
        var speed = new Channel("GPS Speed", new long[] { 0, 1000, 2000, 3000, 4000 }, new double[] { 10, 10, 10, 20, 20 });
        var laps = new List<Lap> { new(1, 1000, 3000, "full"), new(2, 3000, 5000, "full") };

        var distance = SessionLoader.DistancePerLap(speed, laps);

        Assert.Equal(new double[] { 0, 0, 10, 0, 20 }, distance);
    }
}

/// <summary>
/// The libibt path on an iRacing sample. The 70 MB sample is not in this
/// repository (it is libibt's tests/test_data/test.ibt), so the test runs
/// only when INFERNO_IBT_SAMPLE points at it; Fixtures/session_ibt.json is
/// its Python dump (tools/extract_session_fixture.py sample.ibt --out ...).
/// </summary>
public class IbtParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "session_ibt.json")))!;

    private static string SamplePath()
    {
        var path = Environment.GetEnvironmentVariable("INFERNO_IBT_SAMPLE");
        Assert.SkipWhen(string.IsNullOrEmpty(path) || !File.Exists(path), "set INFERNO_IBT_SAMPLE to libibt's tests/test_data/test.ibt");
        Assert.Equal(Fixture["sample"]!.GetValue<string>(), Path.GetFileName(path));
        return path!;
    }

    [Fact]
    public void Native_library_loads()
    {
        Assert.Matches(@"^\d+\.\d+\.\d+", IbtReader.NativeVersion);
    }

    [Fact]
    public void Session_matches_python()
    {
        var raw = SessionLoader.Load(SamplePath(), new ProfileStore("/nonexistent/profiles.yaml"));

        var expectedNames = Fixture["channel_order"]!.AsArray().Select(n => n!.GetValue<string>()).ToList();
        Assert.Equal(expectedNames, raw.ChannelNames.Where(n => n is not ("speed_kmh" or "distance_m")).ToList());
        foreach (var name in expectedNames)
        {
            var expected = Fixture["channels"]![name]!;
            var channel = raw[name];
            Assert.Equal(expected["units"]!.GetValue<string>(), channel.Units);
            Assert.Equal(expected["interpolate"]!.GetValue<bool>(), channel.Interpolate);
            Assert.Equal(expected["values"]!["length"]!.GetValue<int>(), channel.Length);
            var sum = expected["values"]!["sum"]!.GetValue<double>();
            Assert.Equal(sum, channel.Values.Sum(), Math.Max(1e-6, Math.Abs(sum) * 1e-9));
        }

        Assert.Equal("iracing", LogIdentity.GetLoggerId(raw.Metadata));
        Assert.Equal(Fixture["metadata"]!["tick_rate"]!.GetValue<string>(), raw.Metadata["tick_rate"]);
        var expectedLaps = Fixture["session"]!["laps"]!.AsArray().Select(l => l!["num"]!.GetValue<int>()).ToList();
        Assert.Equal(expectedLaps, raw.Laps.Select(l => l.Num).ToList());
        Assert.True(raw.HasChannel("distance_m"));
    }
}
