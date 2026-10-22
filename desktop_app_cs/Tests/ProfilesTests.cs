using System.Text.Json;
using System.Text.Json.Nodes;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Parity of the C# profile loader with the Python package, checked against
/// <c>Fixtures/profiles.json</c> (written by <c>tools/extract_profiles_fixture.py</c>).
/// </summary>
public class ProfilesParityTests
{
    private static readonly JsonNode Fixture = JsonNode.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "profiles.json")))!;

    private static readonly ProfileStore Store = new("/nonexistent/profiles.yaml");

    public static TheoryData<string> BuiltinKeys()
    {
        var data = new TheoryData<string>();
        foreach (var (key, _) in Fixture["builtin"]!.AsObject())
        {
            data.Add(key);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(BuiltinKeys))]
    public void Builtin_profile_matches_python(string key)
    {
        var expected = Fixture["builtin"]![key]!;
        var actual = Store.BuiltinProfiles[key];

        Assert.Equal(expected["name"]!.GetValue<string>(), actual.Name);

        var expectedNames = expected["channel_names"]!.AsObject()
            .ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());
        Assert.Equal(expectedNames, actual.ChannelNames);

        var ratios = expected["motion_ratios"]!;
        Assert.Equal(ratios["front_left"]!.GetValue<double>(), actual.MotionRatios.FrontLeft, 1e-12);
        Assert.Equal(ratios["front_right"]!.GetValue<double>(), actual.MotionRatios.FrontRight, 1e-12);
        Assert.Equal(ratios["rear_left"]!.GetValue<double>(), actual.MotionRatios.RearLeft, 1e-12);
        Assert.Equal(ratios["rear_right"]!.GetValue<double>(), actual.MotionRatios.RearRight, 1e-12);

        if (expected["shock_calibration"] is { } cal)
        {
            Assert.NotNull(actual.ShockCalibration);
            AssertAxle(cal["front"]!, actual.ShockCalibration!.Front);
            AssertAxle(cal["rear"]!, actual.ShockCalibration.Rear);
        }
        else
        {
            Assert.Null(actual.ShockCalibration);
        }

        if (expected["track_widths"] is { } tw)
        {
            Assert.NotNull(actual.TrackWidths);
            Assert.Equal(tw["front_mm"]!.GetValue<double>(), actual.TrackWidths!.FrontMm);
            Assert.Equal(tw["rear_mm"]!.GetValue<double>(), actual.TrackWidths.RearMm);
        }
        else
        {
            Assert.Null(actual.TrackWidths);
        }
    }

    private static void AssertAxle(JsonNode expected, AxleShockCalibration actual)
    {
        Assert.Equal(expected["wheel_per_sensor"]!.GetValue<double>(), actual.WheelPerSensor);
        Assert.Equal(expected["damper_per_sensor"]!.GetValue<double>(), actual.DamperPerSensor);
        Assert.Equal(expected["damper_per_wheel_fit"]?.GetValue<double>(), actual.DamperPerWheelFit);
        Assert.Equal(expected["pot_per_wheel"]!.GetValue<double>(), actual.PotPerWheel, 1e-12);
        Assert.Equal(expected["damper_per_wheel"]!.GetValue<double>(), actual.DamperPerWheel, 1e-12);
        Assert.Equal(expected["wheel_r2"]?.GetValue<double>(), actual.WheelR2);
        Assert.Equal(expected["damper_r2"]?.GetValue<double>(), actual.DamperR2);
        Assert.Equal(expected["damper_wheel_r2"]?.GetValue<double>(), actual.DamperWheelR2);

        var points = expected["points"]!.AsArray()
            .Select(row => row!.AsArray().Select(v => v!.GetValue<double>()).ToList())
            .ToList();
        Assert.Equal(points, (actual.Points ?? Array.Empty<IReadOnlyList<double>>()).Select(r => r.ToList()).ToList());
        var excluded = expected["excluded_points"]!.AsArray().Select(v => v!.GetValue<int>()).ToList();
        Assert.Equal(excluded, (actual.ExcludedPoints ?? Array.Empty<int>()).ToList());
    }

    [Fact]
    public void Builtin_key_set_and_definitions_match_python()
    {
        Assert.Equal(
            Fixture["builtin"]!.AsObject().Select(p => p.Key).OrderBy(k => k, StringComparer.Ordinal),
            Store.BuiltinProfiles.Keys.OrderBy(k => k, StringComparer.Ordinal));
        Assert.Equal(
            Fixture["definitions"]!.AsArray().Select(v => v!.GetValue<string>()),
            Store.LoadProfileDefinitions().Keys.OrderBy(k => k, StringComparer.Ordinal));
    }

    [Fact]
    public void Lookups_match_python()
    {
        foreach (var c in Fixture["lookups"]!.AsArray())
        {
            var loggerId = c!["logger_id"]!.GetValue<string>();
            var vehicle = c["vehicle"]?.GetValue<string>();
            var expected = c["name"]?.GetValue<string>();

            Assert.Equal(expected, Store.GetProfileForLogger(loggerId, vehicle)?.Name);
        }
    }

    [Fact]
    public void Vehicle_name_and_logger_id_match_python()
    {
        foreach (var c in Fixture["vehicle_names"]!.AsArray())
        {
            var metadata = c!["metadata"]!.AsObject().ToDictionary(p => p.Key, p => p.Value!.GetValue<string>());
            var fileName = c["file_name"]!.GetValue<string>();

            Assert.Equal(c["logger_id"]?.GetValue<string>(), LogIdentity.GetLoggerId(metadata));
            Assert.Equal(c["vehicle"]?.GetValue<string>(), LogIdentity.GetVehicleName(metadata, fileName));
            Assert.Equal(c["profile"]?.GetValue<string>(), Store.GetProfileForLog(metadata, fileName)?.Name);
        }
    }
}

/// <summary>Behaviour of the YAML layer itself, mirroring tests/test_profiles.py.</summary>
public class ProfilesYamlTests
{
    private const string TwoCars = """
        profiles:
          car_a:
            name: Car A
            channel_names: {throttle: A_TPS}
          car_b:
            name: Car B
            channel_names: {throttle: B_TPS}
        logger_map:
          8401203:
            default: car_a
            by_vehicle:
              KK-SII: car_a
              KK-F: car_b
        """;

    [Fact]
    public void Nested_entry_resolves_default_and_vehicles()
    {
        var parsed = ProfilesYaml.Parse(TwoCars);

        Assert.Equal("Car A", parsed.Resolved["8401203"].Name);
        Assert.Equal("Car A", parsed.Resolved["8401203/KK-SII"].Name);
        Assert.Equal("Car B", parsed.Resolved["8401203/KK-F"].Name);
        Assert.Equal("A_TPS", parsed.Definitions["car_a"].ChannelNames["throttle"]);
        Assert.Equal("BrakePress", parsed.Definitions["car_a"].ChannelNames["brake"]); // default filled in
    }

    [Fact]
    public void Nested_entry_without_default()
    {
        var parsed = ProfilesYaml.Parse("""
            profiles:
              car_b: {name: Car B}
            logger_map:
              8401203:
                by_vehicle: {KK-F: car_b}
            """);

        Assert.False(parsed.Resolved.ContainsKey("8401203"));
        Assert.Equal("Car B", parsed.Resolved["8401203/KK-F"].Name);
    }

    [Theory]
    [InlineData("")]
    [InlineData("- just\n- a list\n")]
    [InlineData("profiles: [not a mapping]\nlogger_map: 3\n")]
    [InlineData("{unbalanced: [")]
    public void Unreadable_input_is_empty(string yaml)
    {
        var parsed = ProfilesYaml.Parse(yaml);

        Assert.Empty(parsed.Definitions);
        Assert.Empty(parsed.Resolved);
    }

    [Fact]
    public void Motion_ratios_derive_from_calibration_and_explicit_entries_override()
    {
        var parsed = ProfilesYaml.Parse("""
            profiles:
              cal:
                name: Cal
                motion_ratios: {front_left: 0.9}
                shock_calibration:
                  front: {wheel_per_sensor: 1.25, damper_per_sensor: 1.0}
                  rear: {wheel_per_sensor: 2.0, damper_per_sensor: 1.5}
            """);
        var profile = parsed.Definitions["cal"];

        Assert.Equal(0.9, profile.MotionRatios.FrontLeft, 1e-12);
        Assert.Equal(0.8, profile.MotionRatios.FrontRight, 1e-12);
        Assert.Equal(0.5, profile.MotionRatios.RearLeft, 1e-12);
        Assert.Equal(0.8, profile.ShockCalibration!.Front.DamperPerWheel, 1e-12);
        Assert.Equal(0.75, profile.ShockCalibration.Rear.DamperPerWheel, 1e-12);
    }

    [Fact]
    public void Invalid_calibration_is_ignored_and_defaults_apply()
    {
        var parsed = ProfilesYaml.Parse("""
            profiles:
              bad:
                name: Bad
                shock_calibration:
                  front: {wheel_per_sensor: 1.0}
                track_width_mm: {front: -1, rear: 1500}
            """);
        var profile = parsed.Definitions["bad"];

        Assert.Null(profile.ShockCalibration);
        Assert.Null(profile.TrackWidths);
        Assert.Equal(0.997, profile.MotionRatios.FrontLeft, 1e-12);
        Assert.Equal(0.768, profile.MotionRatios.RearRight, 1e-12);
    }

    [Fact]
    public void Write_then_parse_round_trips_including_vehicle_entries()
    {
        var store = new ProfileStore("/nonexistent/profiles.yaml");
        var kkF = store.BuiltinProfiles["8401203/KK-F"];
        var profiles = new Dictionary<string, VehicleProfile>
        {
            ["8401203"] = store.BuiltinProfiles["8401203"],
            ["8401203/KK-F"] = kkF,
            ["6701209"] = store.BuiltinProfiles["6701209"],
        };

        var yaml = ProfilesYaml.Write(profiles);
        var parsed = ProfilesYaml.Parse(yaml);

        Assert.Equal("KK-SII", parsed.Resolved["8401203"].Name);
        Assert.Equal("KK-F", parsed.Resolved["8401203/KK-F"].Name);
        Assert.Equal("Toyota 86 ZN6", parsed.Resolved["6701209"].Name);
        var cal = parsed.Resolved["8401203/KK-F"].ShockCalibration!;
        Assert.Equal(kkF.ShockCalibration!.Front, cal.Front with { Points = kkF.ShockCalibration.Front.Points, ExcludedPoints = kkF.ShockCalibration.Front.ExcludedPoints });
        Assert.Equal(kkF.ShockCalibration.Front.Points!.Select(r => r.ToList()), cal.Front.Points!.Select(r => r.ToList()));
        Assert.Equal(kkF.ShockCalibration.Rear.ExcludedPoints, cal.Rear.ExcludedPoints);
        Assert.Equal(kkF.TrackWidths, parsed.Resolved["8401203/KK-F"].TrackWidths);
        Assert.Equal(kkF.MotionRatios, parsed.Resolved["8401203/KK-F"].MotionRatios);
    }

    [Fact]
    public void Store_prefers_user_profiles_and_saves_merging()
    {
        var dir = Directory.CreateTempSubdirectory("inferno-profiles");
        try
        {
            var store = new ProfileStore(Path.Combine(dir.FullName, "cfg", "profiles.yaml"));
            Assert.Empty(store.LoadUserProfiles());

            store.SaveProfileForLogger("8401203", new VehicleProfile("Car A", new Dictionary<string, string> { ["throttle"] = "A" }));
            store.SaveProfileForLogger("8401203", new VehicleProfile("Car B", new Dictionary<string, string> { ["throttle"] = "B" }), "KK-F");

            Assert.Equal("Car A", store.GetProfileForLogger("8401203")!.Name);
            Assert.Equal("Car B", store.GetProfileForLogger("8401203", "kk-f")!.Name);
            Assert.Equal("Car A", store.GetProfileForLogger("8401203", "unknown car")!.Name);
            Assert.Equal("Toyota 86 ZN6", store.GetProfileForLogger("6701209")!.Name); // builtin fallback
            Assert.Contains("car_b", store.LoadProfileDefinitions().Keys);
            Assert.Contains("kk_f", store.LoadProfileDefinitions().Keys);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
