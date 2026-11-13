using System.Text.Json.Nodes;
using Apache.Arrow;
using Apache.Arrow.Types;
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

    /// <summary>libxrk parses arbitrary bytes as an empty session; the loader refuses it.</summary>
    [Fact]
    public void Load_session_rejects_a_file_with_no_channels()
    {
        var e = Assert.Throws<InvalidDataException>(() => SessionLoader.Load(new byte[] { 1, 2, 3 }, "junk.xrz", BuiltinOnly));
        Assert.Contains("No channels found in junk.xrz", e.Message);
    }

    /// <summary>Same channels in the same order as Python (both sort by channel index).</summary>
    [Fact]
    public void Channel_order_matches_python()
    {
        var expected = Fixture["channel_order"]!.AsArray().Select(n => n!.GetValue<string>());
        Assert.Equal(expected, Raw.Value.ChannelNames);
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
        ParityAssert.ChannelMatches(Fixture["channels"]![name]!, Raw.Value[name], ParityAssert.ValueTolerance(name));
    }

    [Fact]
    public void Raw_laps_match_python()
    {
        ParityAssert.RawLapsMatch(Fixture, Raw.Value);
    }

    [Fact]
    public void Metadata_matches_python()
    {
        ParityAssert.MetadataMatches(Fixture, Raw.Value);
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

        ParityAssert.SummaryMatches(expected["speed_kmh"]!, session["speed_kmh"].Values);
        ParityAssert.SummaryMatches(expected["distance_m"]!, session["distance_m"].Values);

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
        var fixture = InfernoAnalyzer.App.Fixture.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "laps.json"));
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

/// <summary>
/// Comparisons against a <c>tools/extract_session_fixture.py</c> dump. Each
/// fixture comes from the parser's Python package at the version the FFI
/// library is built from, so the samples are bit-identical.
/// </summary>
internal static class ParityAssert
{
    /// <summary>
    /// How far a channel's values may be from the fixture, which was written on Linux. Zero for every
    /// channel but GPS Altitude: the parser derives it from ECEF coordinates through the C runtime's
    /// trigonometry, as a difference of Earth-radius-sized terms, so a last-bit difference between C
    /// runtimes shows up around 1e-9 m (one ulp at 6.4e6 m is 9e-10 m). Seen on Windows (MSVC):
    /// 1.4e-9 m. Latitude and longitude come out of the same conversion bit for bit.
    /// </summary>
    public static double ValueTolerance(string channelName) => channelName == "GPS Altitude" ? 1e-8 : 0;

    /// <summary>Units, decimals, interpolation flag, and the timecode and value summaries of one channel.</summary>
    public static void ChannelMatches(JsonNode expected, Channel channel, double valueTolerance = 0)
    {
        Assert.Equal(expected["units"]!.GetValue<string>(), channel.Units);
        Assert.Equal(expected["dec_pts"]!.GetValue<int>(), channel.DecPts);
        Assert.Equal(expected["interpolate"]!.GetValue<bool>(), channel.Interpolate);
        SummaryMatches(expected["timecodes"]!, channel.Timecodes.Select(t => (double)t).ToArray());
        SummaryMatches(expected["values"]!, channel.Values, valueTolerance);
    }

    /// <summary>
    /// Length, first, last, min and max exactly (or within <paramref name="tolerance"/> when one is given);
    /// the sum to 1e-9 relative (numpy sums pairwise, LINQ left to right).
    /// </summary>
    public static void SummaryMatches(JsonNode expected, double[] values, double tolerance = 0)
    {
        Assert.Equal(expected["length"]!.GetValue<int>(), values.Length);
        if (values.Length == 0)
        {
            return;
        }

        Assert.Equal(expected["first"]!.GetValue<double>(), values[0], tolerance);
        Assert.Equal(expected["last"]!.GetValue<double>(), values[^1], tolerance);
        Assert.Equal(expected["min"]!.GetValue<double>(), values.Min(), tolerance);
        Assert.Equal(expected["max"]!.GetValue<double>(), values.Max(), tolerance);
        var sum = expected["sum"]!.GetValue<double>();
        Assert.Equal(sum, values.Sum(), Math.Max(1e-9, Math.Abs(sum) * 1e-9));
    }

    /// <summary>The laps table as the parser returns it, before cleaning.</summary>
    public static void RawLapsMatch(JsonNode fixture, LogFile log)
    {
        var expected = fixture["laps_raw"]!.AsArray()
            .Select(l => new Lap(l!["num"]!.GetValue<int>(), l["start_time"]!.GetValue<long>(), l["end_time"]!.GetValue<long>(), l["lap_type"]!.GetValue<string>(), l["session"]?.GetValue<int>()))
            .ToList();
        Assert.Equal(expected, log.Laps);
    }

    /// <summary>
    /// The metadata document: Python's keys in Python's order, scalar
    /// entries as the text Python's <c>str()</c> gives, nested entries
    /// (lists and mappings) equal as JSON.
    /// </summary>
    public static void MetadataMatches(JsonNode fixture, LogFile log)
    {
        var keys = fixture["metadata_keys"]!.AsArray().Select(k => k!.GetValue<string>()).ToList();
        Assert.Equal(keys, JsonNode.Parse(log.MetadataJson)!.AsObject().Select(p => p.Key).ToList());
        Assert.Equal(keys.Order(StringComparer.Ordinal), log.Metadata.Keys.Order(StringComparer.Ordinal));

        foreach (var (key, value) in fixture["metadata"]!.AsObject())
        {
            Assert.Equal(value!.GetValue<string>(), log.Metadata[key]);
        }

        foreach (var (key, value) in fixture["metadata_nested"]!.AsObject())
        {
            var expected = JsonNode.Parse(value!.GetValue<string>());
            Assert.True(JsonNode.DeepEquals(expected, JsonNode.Parse(log.Metadata[key])), $"metadata {key} differs from Python");
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
/// repository (it is libibt's tests/test_data/test.ibt), so the tests that
/// read it run only when INFERNO_IBT_SAMPLE points at it (CI uses the copy
/// in the libibt source fetched by tools/build_native.py);
/// Fixtures/session_ibt.json is its dump by the libibt release the project
/// depends on, which is the release the FFI library is built from
/// (tools/extract_session_fixture.py sample.ibt --out ...).
/// </summary>
public class IbtParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "session_ibt.json")))!;

    private static readonly Lazy<LogFile> Sample = new(() => IbtReader.Read(Environment.GetEnvironmentVariable("INFERNO_IBT_SAMPLE")!));

    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");

    /// <summary>The parsed sample, or a skip when it is not available.</summary>
    private static LogFile Raw()
    {
        var path = Environment.GetEnvironmentVariable("INFERNO_IBT_SAMPLE");
        Assert.SkipWhen(string.IsNullOrEmpty(path) || !File.Exists(path), "set INFERNO_IBT_SAMPLE to libibt's tests/test_data/test.ibt");
        Assert.Equal(Fixture["sample"]!.GetValue<string>(), Path.GetFileName(path));
        return Sample.Value;
    }

    [Fact]
    public void Native_library_loads()
    {
        Assert.Matches(@"^\d+\.\d+\.\d+", IbtReader.NativeVersion);
    }

    /// <summary>The parser's own message comes back, for a pointer-less empty buffer too.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void Bytes_that_are_not_an_ibt_file_are_rejected(int length)
    {
        var e = Assert.Throws<InvalidDataException>(() => IbtReader.Read(new byte[length], "junk.ibt"));
        Assert.DoesNotContain("NULL", e.Message);
        Assert.NotEqual("libibt could not read junk.ibt", e.Message);
    }

    [Fact]
    public void Channels_match_python()
    {
        var raw = Raw();

        var expectedNames = Fixture["channel_order"]!.AsArray().Select(n => n!.GetValue<string>()).ToList();
        Assert.Equal(expectedNames, raw.ChannelNames);
        foreach (var name in expectedNames)
        {
            ParityAssert.ChannelMatches(Fixture["channels"]![name]!, raw[name], ParityAssert.ValueTolerance(name));
        }
    }

    /// <summary>
    /// A time-subsample array variable is one channel at its own rate: six
    /// samples per 60 Hz tick, on timecodes of its own.
    /// </summary>
    [Fact]
    public void Time_subsample_channel_is_read_at_360_hz()
    {
        var raw = Raw();

        var torque = raw["SteeringWheelTorque_ST"];
        Assert.Equal(6 * raw["SteeringWheelTorque"].Length, torque.Length);
        Assert.Equal(ChannelValueType.Float32, torque.ValueType);
        Assert.True(torque.Interpolate);
        Assert.Equal("N*m", torque.Units);
        Assert.Equal(1000.0 / 360, (torque.Timecodes[^1] - torque.Timecodes[0]) / (double)(torque.Length - 1), 1e-3);
    }

    [Fact]
    public void Raw_laps_match_python()
    {
        ParityAssert.RawLapsMatch(Fixture, Raw());
    }

    /// <summary>
    /// The whole <c>LogFile.metadata</c> dict crosses the boundary: header
    /// fields, the session-info YAML, its parse and the entries derived from it.
    /// </summary>
    [Fact]
    public void Metadata_matches_python()
    {
        var raw = Raw();

        ParityAssert.MetadataMatches(Fixture, raw);
        Assert.Equal("iracing", LogIdentity.GetLoggerId(raw.Metadata));
        Assert.Equal("iRacing", BuiltinOnly.GetProfileForLog(raw.Metadata, raw.FileName)?.Name);

        var sessionInfo = JsonNode.Parse(raw.Metadata["session_info"])!;
        Assert.Equal(raw.Metadata["track_name"], sessionInfo["WeekendInfo"]!["TrackName"]!.GetValue<string>());
        var driver = sessionInfo["DriverInfo"]!["Drivers"]!.AsArray()
            .Single(d => d!["CarIdx"]!.GetValue<int>() == sessionInfo["DriverInfo"]!["DriverCarIdx"]!.GetValue<int>())!;
        Assert.Equal(raw.Metadata["driver_name"], driver["UserName"]!.GetValue<string>());
        Assert.Equal(raw.Metadata["car_name"], driver["CarScreenName"]!.GetValue<string>());
    }

    [Fact]
    public void Load_session_derived_channels_and_laps_match_python()
    {
        var session = SessionLoader.PostProcess(Raw(), BuiltinOnly, SessionLoader.DefaultLapTypes);
        var expected = Fixture["session"]!;

        var laps = expected["laps"]!.AsArray().Select(l => (l!["num"]!.GetValue<int>(), l["lap_time_s"]!.GetValue<double>())).ToList();
        Assert.Equal(laps, session.Laps.Select(l => (l.Num, l.LapTimeS)).ToList());
        ParityAssert.SummaryMatches(expected["speed_kmh"]!, session["speed_kmh"].Values);
        ParityAssert.SummaryMatches(expected["distance_m"]!, session["distance_m"].Values);
    }
}

/// <summary>Arrow batches → channels, on batches built by hand.</summary>
public class ArrowConvertTests
{
    /// <summary>A fresh column per batch: disposing a batch releases its arrays.</summary>
    private static Int64Array Timecodes() => new Int64Array.Builder().AppendRange(new long[] { 0, 16, 33 }).Build();

    [Fact]
    public void Scalar_column_becomes_a_channel_with_its_field_metadata()
    {
        var values = new FloatArray.Builder().AppendRange(new[] { 1.5f, 2.5f, 3.5f }).Build();
        var schema = new Schema.Builder()
            .Field(f => f.Name("timecodes").DataType(Int64Type.Default))
            .Field(f => f.Name("Speed").DataType(FloatType.Default).Metadata("units", "m/s").Metadata("interpolate", "True"))
            .Build();
        using var batch = new RecordBatch(schema, new IArrowArray[] { Timecodes(), values }, 3);

        var channel = ArrowConvert.ToChannel(batch, "Speed");

        Assert.NotNull(channel);
        Assert.Equal(new long[] { 0, 16, 33 }, channel.Timecodes);
        Assert.Equal(new[] { 1.5, 2.5, 3.5 }, channel.Values);
        Assert.Equal(ChannelValueType.Float32, channel.ValueType);
        Assert.Equal("m/s", channel.Units);
        Assert.True(channel.Interpolate);
    }

    /// <summary>libibt's per-car array variables: one FixedSizeList per tick.</summary>
    [Fact]
    public void Vector_column_is_not_a_channel()
    {
        var builder = new FixedSizeListArray.Builder(Int32Type.Default, 2);
        var items = (Int32Array.Builder)builder.ValueBuilder;
        for (var tick = 0; tick < 3; tick++)
        {
            builder.Append();
            items.Append(tick).Append(-tick);
        }

        var vectors = builder.Build();
        var schema = new Schema.Builder()
            .Field(f => f.Name("timecodes").DataType(Int64Type.Default))
            .Field(f => f.Name("CarIdxLap").DataType(vectors.Data.DataType))
            .Build();
        using var batch = new RecordBatch(schema, new IArrowArray[] { Timecodes(), vectors }, 3);

        Assert.Null(ArrowConvert.ToChannel(batch, "CarIdxLap"));
    }
}
