"""Tests for profiles module."""

import importlib.resources
from pathlib import Path

import pytest
import yaml

from helpers import MockLogFile
from motorsports_data_notebook._util import validate_channel_names
from motorsports_data_notebook.profiles import (
    ALL_CANONICAL_KEYS,
    DEFAULT_CHANNEL_NAMES,
    VehicleProfile,
    _get_user_profiles_path,
    _load_profiles_file,
    _profile_from_dict,
    _profile_to_dict,
    get_logger_id,
    get_profile_for_log,
    get_profile_for_logger,
    get_vehicle_name,
    is_iracing_session,
    load_builtin_profiles,
    load_profile_definitions,
    load_user_profiles,
    profile_key,
    save_profile_for_logger,
    save_user_profiles,
    split_profile_key,
)
from motorsports_data_notebook.suspension import MotionRatios


class TestVehicleProfile:
    """Tests for VehicleProfile dataclass."""

    def test_default_values(self):
        """Default profile should have all canonical keys with default channel names."""
        profile = VehicleProfile(name="Test")
        assert profile.channel_names == DEFAULT_CHANNEL_NAMES
        assert profile.motion_ratios.front_left == pytest.approx(0.997)
        assert profile.motion_ratios.rear_left == pytest.approx(0.768)

    def test_custom_values(self):
        """Should accept custom channel names and motion ratios."""
        channels = {"throttle": "TPS", "brake": "BP"}
        ratios = MotionRatios(front_left=0.9, front_right=0.9, rear_left=0.8, rear_right=0.8)
        profile = VehicleProfile(name="Custom", channel_names=channels, motion_ratios=ratios)
        assert profile.channel_names["throttle"] == "TPS"
        assert profile.motion_ratios.front_left == pytest.approx(0.9)

    def test_get_channel_subset(self):
        """get_channel_subset should return only requested keys."""
        profile = VehicleProfile(name="Test")
        subset = profile.get_channel_subset(["throttle", "brake"])
        assert subset == {"throttle": "PPS", "brake": "BrakePress"}
        assert "shock_fl" not in subset

    def test_get_channel_subset_missing_keys(self):
        """get_channel_subset should skip keys not in channel_names."""
        profile = VehicleProfile(name="Test", channel_names={"throttle": "PPS"})
        subset = profile.get_channel_subset(["throttle", "nonexistent"])
        assert subset == {"throttle": "PPS"}


class TestValidateChannelNames:
    """Tests for validate_channel_names."""

    def test_valid_keys(self):
        """Should not raise when all required keys are present."""
        validate_channel_names(
            {"throttle": "PPS", "brake": "BP"}, ["throttle", "brake"], "test_func"
        )

    def test_missing_keys(self):
        """Should raise KeyError listing missing keys."""
        with pytest.raises(KeyError, match="Missing.*brake"):
            validate_channel_names({"throttle": "PPS"}, ["throttle", "brake"], "test_func")


class TestProfileSerialization:
    """Tests for profile serialization roundtrip."""

    def test_roundtrip(self):
        """Serializing and deserializing should produce an equivalent profile."""
        original = VehicleProfile(
            name="Test Car",
            channel_names=DEFAULT_CHANNEL_NAMES.copy(),
            motion_ratios=MotionRatios(
                front_left=0.95, front_right=0.95, rear_left=0.75, rear_right=0.75
            ),
        )
        data = _profile_to_dict(original)
        restored = _profile_from_dict(data)

        assert restored.name == original.name
        assert restored.channel_names == original.channel_names
        assert restored.motion_ratios.front_left == pytest.approx(original.motion_ratios.front_left)
        assert restored.motion_ratios.rear_left == pytest.approx(original.motion_ratios.rear_left)

    def test_missing_keys_use_defaults(self):
        """Deserializing with missing channel keys should fill from defaults."""
        data = {
            "name": "Partial",
            "channel_names": {"throttle": "CustomTPS"},
        }
        profile = _profile_from_dict(data)
        # Custom key preserved
        assert profile.channel_names["throttle"] == "CustomTPS"
        # Missing keys filled from defaults
        assert profile.channel_names["brake"] == DEFAULT_CHANNEL_NAMES["brake"]
        assert profile.channel_names["shock_fl"] == DEFAULT_CHANNEL_NAMES["shock_fl"]

    def test_missing_motion_ratios_use_defaults(self):
        """Deserializing without motion_ratios should use defaults."""
        data = {"name": "NoRatios"}
        profile = _profile_from_dict(data)
        assert profile.motion_ratios.front_left == pytest.approx(0.997)
        assert profile.motion_ratios.rear_left == pytest.approx(0.768)

    def test_forward_compatible_extra_keys(self):
        """Extra keys in channel_names should be preserved through roundtrip."""
        data = {
            "name": "Future",
            "channel_names": {
                "throttle": "PPS",
                "new_future_key": "FutureChannel",
            },
        }
        profile = _profile_from_dict(data)
        assert profile.channel_names["new_future_key"] == "FutureChannel"


class TestLoadProfilesFile:
    """Tests for _load_profiles_file."""

    def test_valid_file(self, tmp_path):
        """Should parse a valid two-tier YAML file."""
        yaml_content = {
            "profiles": {
                "car_a": {
                    "name": "Car A",
                    "channel_names": {"throttle": "TPS_A"},
                    "motion_ratios": {
                        "front_left": 0.9,
                        "front_right": 0.9,
                        "rear_left": 0.8,
                        "rear_right": 0.8,
                    },
                },
            },
            "logger_map": {"1234": "car_a", "5678": "car_a"},
        }
        path = tmp_path / "profiles.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        result = _load_profiles_file(path)
        assert "1234" in result
        assert "5678" in result
        assert result["1234"].name == "Car A"
        assert result["1234"].channel_names["throttle"] == "TPS_A"
        assert result["1234"].motion_ratios.front_left == pytest.approx(0.9)

    def test_missing_file(self, tmp_path):
        """Should return empty dict for missing file."""
        result = _load_profiles_file(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_corrupt_file(self, tmp_path):
        """Should return empty dict for corrupt YAML."""
        path = tmp_path / "bad.yaml"
        path.write_text("{{not valid yaml::")
        result = _load_profiles_file(path)
        assert result == {}

    def test_empty_file(self, tmp_path):
        """Should return empty dict for empty file."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = _load_profiles_file(path)
        assert result == {}

    def test_unmapped_logger(self, tmp_path):
        """Logger IDs pointing to nonexistent profiles should be skipped."""
        yaml_content = {
            "profiles": {"car_a": {"name": "Car A"}},
            "logger_map": {"1234": "car_b"},  # car_b doesn't exist
        }
        path = tmp_path / "profiles.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        result = _load_profiles_file(path)
        assert "1234" not in result

    def test_numeric_logger_id(self, tmp_path):
        """Numeric logger IDs in YAML should be converted to strings."""
        yaml_content = {
            "profiles": {"car_a": {"name": "Car A"}},
            "logger_map": {12345678: "car_a"},
        }
        path = tmp_path / "profiles.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        result = _load_profiles_file(path)
        assert "12345678" in result


class TestUserProfiles:
    """Tests for user profile save/load."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Saving and loading profiles should roundtrip."""
        user_path = tmp_path / "config" / "motorsports_data_notebook" / "profiles.yaml"
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: user_path,
        )

        profile = VehicleProfile(
            name="My Car",
            channel_names={"throttle": "MyTPS", "brake": "MyBP"},
            motion_ratios=MotionRatios(
                front_left=0.95, front_right=0.95, rear_left=0.75, rear_right=0.75
            ),
        )

        save_user_profiles({"ABCD1234": profile})
        loaded = load_user_profiles()

        assert "ABCD1234" in loaded
        assert loaded["ABCD1234"].name == "My Car"
        assert loaded["ABCD1234"].channel_names["throttle"] == "MyTPS"
        assert loaded["ABCD1234"].motion_ratios.front_left == pytest.approx(0.95)

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        """Loading from nonexistent path should return empty dict."""
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: tmp_path / "nonexistent" / "profiles.yaml",
        )
        result = load_user_profiles()
        assert result == {}

    def test_corrupt_file_returns_empty(self, tmp_path, monkeypatch):
        """Loading from corrupt file should return empty dict."""
        user_path = tmp_path / "profiles.yaml"
        user_path.write_text("{{invalid yaml::")
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: user_path,
        )
        result = load_user_profiles()
        assert result == {}

    def test_save_multiple_loggers_same_profile(self, tmp_path, monkeypatch):
        """Multiple loggers sharing a profile should not duplicate profile definitions."""
        user_path = tmp_path / "config" / "motorsports_data_notebook" / "profiles.yaml"
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: user_path,
        )

        profile = VehicleProfile(name="Shared Car")
        save_user_profiles({"AAA": profile, "BBB": profile})

        # Read raw YAML to verify dedup
        with open(user_path) as f:
            raw = yaml.safe_load(f)
        assert len(raw["profiles"]) == 1
        assert len(raw["logger_map"]) == 2


class TestGetProfileForLogger:
    """Tests for get_profile_for_logger."""

    def test_unknown_logger(self, tmp_path, monkeypatch):
        """Unknown logger ID should return None."""
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: tmp_path / "nonexistent.yaml",
        )
        result = get_profile_for_logger("unknown_id")
        assert result is None

    def test_user_overrides_builtin(self, tmp_path, monkeypatch):
        """User profile should take precedence over builtin."""
        user_path = tmp_path / "profiles.yaml"
        user_profile = VehicleProfile(name="User Car", channel_names={"throttle": "UserTPS"})

        # Save user profile
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: user_path,
        )
        save_user_profiles({"LOGGER1": user_profile})

        # Also make builtin return a different profile for same logger
        builtin_profile = VehicleProfile(
            name="Builtin Car", channel_names={"throttle": "BuiltinTPS"}
        )
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles.load_builtin_profiles",
            lambda: {"LOGGER1": builtin_profile},
        )

        result = get_profile_for_logger("LOGGER1")
        assert result is not None
        assert result.name == "User Car"

    def test_falls_back_to_builtin(self, tmp_path, monkeypatch):
        """Should fall back to builtin when user has no match."""
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: tmp_path / "nonexistent.yaml",
        )
        builtin_profile = VehicleProfile(name="Builtin Car")
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles.load_builtin_profiles",
            lambda: {"LOGGER1": builtin_profile},
        )

        result = get_profile_for_logger("LOGGER1")
        assert result is not None
        assert result.name == "Builtin Car"


class TestSaveProfileForLogger:
    """Tests for save_profile_for_logger."""

    def test_merges_with_existing(self, tmp_path, monkeypatch):
        """Saving a profile should merge with existing profiles."""
        user_path = tmp_path / "config" / "motorsports_data_notebook" / "profiles.yaml"
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path",
            lambda: user_path,
        )

        # Save first profile
        profile1 = VehicleProfile(name="Car 1")
        save_profile_for_logger("AAA", profile1)

        # Save second profile
        profile2 = VehicleProfile(name="Car 2")
        save_profile_for_logger("BBB", profile2)

        # Both should be present
        loaded = load_user_profiles()
        assert "AAA" in loaded
        assert "BBB" in loaded
        assert loaded["AAA"].name == "Car 1"
        assert loaded["BBB"].name == "Car 2"


class TestGetLoggerId:
    """Tests for get_logger_id."""

    def test_extraction_from_metadata(self):
        """Should extract Logger ID from metadata dict."""
        log = MockLogFile(metadata={"Logger ID": "12345678"})
        assert get_logger_id(log) == "12345678"

    def test_alternative_key(self):
        """Should try logger_id as fallback key."""
        log = MockLogFile(metadata={"logger_id": "87654321"})
        assert get_logger_id(log) == "87654321"

    def test_missing_metadata(self):
        """Should return None when metadata has no logger ID."""
        log = MockLogFile(metadata={"other_key": "value"})
        assert get_logger_id(log) is None

    def test_no_metadata_attr(self):
        """Should return None when object has no metadata attribute."""

        class NoMetadata:
            pass

        assert get_logger_id(NoMetadata()) is None

    def test_none_metadata(self):
        """Should return None when metadata is None."""
        log = MockLogFile(metadata=None)
        # metadata is set to {} by __post_init__, so test with raw object
        log.metadata = None
        assert get_logger_id(log) is None


class TestBuiltinProfiles:
    """Tests for builtin profiles file."""

    def test_builtin_loads(self):
        """Builtin profiles YAML should load and resolve logger mappings."""
        result = load_builtin_profiles()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_builtin_yaml_valid(self):
        """Builtin YAML should be parseable and have expected structure."""
        ref = importlib.resources.files("motorsports_data_notebook.data").joinpath(
            "builtin_profiles.yaml"
        )
        with importlib.resources.as_file(ref) as path:
            with open(path) as f:
                data = yaml.safe_load(f)

        assert "profiles" in data
        assert "logger_map" in data

        # Each profile should have a name and channel_names dict
        for profile_id, profile_data in data["profiles"].items():
            assert "name" in profile_data, f"Profile '{profile_id}' missing 'name'"
            assert (
                "channel_names" in profile_data
            ), f"Profile '{profile_id}' missing 'channel_names'"
            assert isinstance(profile_data["channel_names"], dict)
            # All channel values must be non-empty strings
            for key, value in profile_data["channel_names"].items():
                assert key in ALL_CANONICAL_KEYS, f"Profile '{profile_id}' has unknown key '{key}'"
                assert (
                    isinstance(value, str) and value
                ), f"Profile '{profile_id}' channel '{key}' must be a non-empty string"

        # Every logger_map entry should reference existing profiles. Entries are
        # either a profile id or {default: id, by_vehicle: {vehicle: id}}.
        for logger_id, entry in data["logger_map"].items():
            if isinstance(entry, dict):
                referenced = [entry.get("default")] + list(entry.get("by_vehicle", {}).values())
                referenced = [pid for pid in referenced if pid is not None]
                assert referenced, f"Logger '{logger_id}' maps to no profiles"
            else:
                referenced = [entry]
            for profile_id in referenced:
                assert (
                    profile_id in data["profiles"]
                ), f"Logger '{logger_id}' references unknown profile '{profile_id}'"


class TestGetLoggerIdIRacing:
    """Tests for get_logger_id with iRacing metadata."""

    def test_returns_iracing_for_ibt_metadata(self):
        """iRacing metadata (session_info_yaml) should return 'iracing'."""
        log = MockLogFile(metadata={"session_info_yaml": "some_yaml_content"})
        assert get_logger_id(log) == "iracing"

    def test_aim_logger_id_takes_precedence(self):
        """If both Logger ID and session_info_yaml exist, Logger ID wins."""
        log = MockLogFile(metadata={"Logger ID": "12345", "session_info_yaml": "yaml"})
        assert get_logger_id(log) == "12345"


class TestIsIracingSession:
    """Tests for is_iracing_session function."""

    def test_true_for_iracing(self):
        """Should return True for iRacing metadata."""
        log = MockLogFile(metadata={"session_info_yaml": "some_yaml"})
        assert is_iracing_session(log) is True

    def test_false_for_aim(self):
        """Should return False for AIM metadata."""
        log = MockLogFile(metadata={"Logger ID": "6701209"})
        assert is_iracing_session(log) is False

    def test_false_for_empty_metadata(self):
        """Should return False for empty metadata."""
        log = MockLogFile(metadata={})
        assert is_iracing_session(log) is False

    def test_false_for_no_metadata(self):
        """Should return False when metadata is not a dict."""
        log = MockLogFile()
        log.metadata = None
        assert is_iracing_session(log) is False


class TestBuiltinIracingProfile:
    """Tests for the builtin iRacing profile."""

    def test_iracing_profile_exists(self):
        """The builtin profiles should include an 'iracing' entry."""
        profile = get_profile_for_logger("iracing")
        assert profile is not None

    def test_iracing_profile_name(self):
        """The iRacing profile should have the correct name."""
        profile = get_profile_for_logger("iracing")
        assert profile.name == "iRacing"

    def test_iracing_profile_has_correct_channels(self):
        """The iRacing profile should have correct channel mappings."""
        profile = get_profile_for_logger("iracing")
        assert profile.channel_names["throttle"] == "Throttle"
        assert profile.channel_names["brake"] == "Brake"
        assert profile.channel_names["gps_speed"] == "Speed"
        assert profile.channel_names["gps_latitude"] == "Lat"
        assert profile.channel_names["gps_longitude"] == "Lon"
        assert profile.channel_names["lateral_g"] == "LatAccel"
        assert profile.channel_names["steering"] == "SteeringWheelAngle"

    def test_iracing_profile_has_shock_velocity_channels(self):
        """The iRacing profile should map shock channels to velocity names."""
        profile = get_profile_for_logger("iracing")
        assert profile.channel_names["shock_fl"] == "LFshockVel"
        assert profile.channel_names["shock_fr"] == "RFshockVel"
        assert profile.channel_names["shock_rl"] == "LRshockVel"
        assert profile.channel_names["shock_rr"] == "RRshockVel"

    def test_iracing_profile_motion_ratios_are_unity(self):
        """The iRacing profile should have 1.0 motion ratios."""
        profile = get_profile_for_logger("iracing")
        assert profile.motion_ratios.front_left == pytest.approx(1.0)
        assert profile.motion_ratios.front_right == pytest.approx(1.0)
        assert profile.motion_ratios.rear_left == pytest.approx(1.0)
        assert profile.motion_ratios.rear_right == pytest.approx(1.0)


# ── Shared loggers: vehicle name as second discriminator ─────────────────────


class TestVehicleName:
    """Tests for get_vehicle_name()."""

    def test_metadata_vehicle_wins(self):
        log = MockLogFile(
            metadata={"Logger ID": 8401203, "Vehicle": "KK-F"},
            file_name="CMD_KK-SII_Suzuka Car_Race_a_0358.xrk",
        )
        assert get_vehicle_name(log) == "KK-F"

    def test_blank_metadata_falls_back_to_file_name(self):
        """xrz exports have an empty Vehicle field; use the Race Studio file name."""
        log = MockLogFile(
            metadata={"Logger ID": 8401203, "Vehicle": ""},
            file_name="/data/2026-10-02/CMD_KK-F_Suzuka West_Generic testing_a_0351.xrz",
        )
        assert get_vehicle_name(log) == "KK-F"

    def test_file_name_with_underscored_track(self):
        log = MockLogFile(
            metadata={"Logger ID": 8401203},
            file_name="CMD_KK-F_Tsukuba_Car_Race_a_0344.xrz",
        )
        assert get_vehicle_name(log) == "KK-F"

    def test_explicit_file_name_argument(self):
        """Logs loaded from bytes have no file name; the caller can pass one."""
        log = MockLogFile(metadata={"Logger ID": 8401203}, file_name="")
        assert get_vehicle_name(log, file_name="CMD_KK-SII_Fuji GP_Race_a_0001.xrz") == "KK-SII"

    def test_non_racestudio_name_returns_none(self):
        log = MockLogFile(metadata={"Logger ID": 1}, file_name="session.xrk")
        assert get_vehicle_name(log) is None

    def test_non_aim_file_does_not_parse_name(self):
        log = MockLogFile(metadata={"session_info_yaml": "x"}, file_name="mx5_okayama_2024.ibt")
        assert get_vehicle_name(log) is None

    def test_no_metadata(self):
        assert get_vehicle_name(object()) is None  # type: ignore[arg-type]


class TestProfileKeys:
    def test_profile_key_roundtrip(self):
        assert profile_key("8401203", "KK-F") == "8401203/KK-F"
        assert profile_key("8401203") == "8401203"
        assert split_profile_key("8401203/KK-F") == ("8401203", "KK-F")
        assert split_profile_key("8401203") == ("8401203", None)


class TestVehicleLoggerMap:
    """Nested logger_map entries: default + by_vehicle."""

    def _write(self, tmp_path, logger_entry):
        yaml_content = {
            "profiles": {
                "car_a": {"name": "Car A", "channel_names": {"throttle": "A_TPS"}},
                "car_b": {"name": "Car B", "channel_names": {"throttle": "B_TPS"}},
            },
            "logger_map": {8401203: logger_entry},
        }
        path = tmp_path / "profiles.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)
        return path

    def test_nested_entry_resolves_default_and_vehicles(self, tmp_path):
        path = self._write(
            tmp_path, {"default": "car_a", "by_vehicle": {"KK-SII": "car_a", "KK-F": "car_b"}}
        )
        result = _load_profiles_file(path)
        assert result["8401203"].name == "Car A"
        assert result["8401203/KK-SII"].name == "Car A"
        assert result["8401203/KK-F"].name == "Car B"

    def test_nested_entry_without_default(self, tmp_path):
        path = self._write(tmp_path, {"by_vehicle": {"KK-F": "car_b"}})
        result = _load_profiles_file(path)
        assert "8401203" not in result
        assert result["8401203/KK-F"].name == "Car B"

    def test_get_profile_for_logger_prefers_vehicle(self, tmp_path, monkeypatch):
        path = self._write(
            tmp_path, {"default": "car_a", "by_vehicle": {"KK-SII": "car_a", "KK-F": "car_b"}}
        )
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path", lambda: path
        )
        monkeypatch.setattr("motorsports_data_notebook.profiles.load_builtin_profiles", lambda: {})
        assert get_profile_for_logger("8401203", "KK-F").name == "Car B"
        assert get_profile_for_logger("8401203", "kk-f").name == "Car B"  # case-insensitive
        assert get_profile_for_logger("8401203", "KK-SII").name == "Car A"
        assert get_profile_for_logger("8401203").name == "Car A"
        # Unknown vehicle falls back to the logger default
        assert get_profile_for_logger("8401203", "Something Else").name == "Car A"

    def test_get_profile_for_log_uses_vehicle(self, tmp_path, monkeypatch):
        path = self._write(tmp_path, {"default": "car_a", "by_vehicle": {"KK-F": "car_b"}})
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path", lambda: path
        )
        monkeypatch.setattr("motorsports_data_notebook.profiles.load_builtin_profiles", lambda: {})
        log = MockLogFile(
            metadata={"Logger ID": 8401203, "Vehicle": ""},
            file_name="CMD_KK-F_Suzuka West_Generic testing_a_0351.xrz",
        )
        assert get_profile_for_log(log).name == "Car B"
        log_b = MockLogFile(metadata={"Logger ID": 8401203}, file_name="")
        assert get_profile_for_log(log_b).name == "Car A"
        assert get_profile_for_log(MockLogFile(metadata={})) is None

    def test_save_and_load_vehicle_specific(self, tmp_path, monkeypatch):
        user_path = tmp_path / "config" / "profiles.yaml"
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path", lambda: user_path
        )
        default = VehicleProfile(name="Car A", channel_names={"throttle": "A"})
        kk_f = VehicleProfile(name="Car B", channel_names={"throttle": "B"})
        save_profile_for_logger("8401203", default)
        save_profile_for_logger("8401203", kk_f, vehicle="KK-F")

        with open(user_path) as f:
            raw = yaml.safe_load(f)
        assert raw["logger_map"]["8401203"] == {"default": "car_a", "by_vehicle": {"KK-F": "car_b"}}

        loaded = load_user_profiles()
        assert loaded["8401203"].name == "Car A"
        assert loaded["8401203/KK-F"].name == "Car B"

    def test_load_profile_definitions_keyed_by_id(self, tmp_path, monkeypatch):
        path = self._write(tmp_path, "car_a")
        monkeypatch.setattr(
            "motorsports_data_notebook.profiles._get_user_profiles_path", lambda: path
        )
        defs = load_profile_definitions()
        assert defs["car_a"].name == "Car A"
        assert defs["car_b"].name == "Car B"
        assert "kk_f" in defs  # builtin definitions are included too


class TestShockCalibrationProfiles:
    """shock_calibration section: slopes kept, motion ratios derived."""

    def test_motion_ratios_derived_from_calibration(self):
        profile = _profile_from_dict(
            {
                "name": "Cal",
                "shock_calibration": {
                    "front": {"wheel_per_sensor": 1.25, "damper_per_sensor": 1.0},
                    "rear": {"wheel_per_sensor": 2.0, "damper_per_sensor": 1.5},
                },
            }
        )
        assert profile.shock_calibration is not None
        assert profile.motion_ratios.front_left == pytest.approx(0.8)
        assert profile.motion_ratios.front_right == pytest.approx(0.8)
        assert profile.motion_ratios.rear_left == pytest.approx(0.5)
        assert profile.shock_calibration.front.pot_per_wheel == pytest.approx(0.8)
        assert profile.shock_calibration.front.damper_per_wheel == pytest.approx(0.8)
        assert profile.shock_calibration.rear.damper_per_wheel == pytest.approx(0.75)

    def test_explicit_motion_ratios_override_calibration(self):
        profile = _profile_from_dict(
            {
                "name": "Cal",
                "motion_ratios": {"front_left": 0.9},
                "shock_calibration": {
                    "front": {"wheel_per_sensor": 1.25, "damper_per_sensor": 1.0},
                    "rear": {"wheel_per_sensor": 2.0, "damper_per_sensor": 1.5},
                },
            }
        )
        assert profile.motion_ratios.front_left == pytest.approx(0.9)
        assert profile.motion_ratios.front_right == pytest.approx(0.8)  # still derived

    def test_invalid_calibration_ignored(self):
        profile = _profile_from_dict(
            {"name": "Bad", "shock_calibration": {"front": {"wheel_per_sensor": 1.0}}}
        )
        assert profile.shock_calibration is None
        assert profile.motion_ratios.front_left == pytest.approx(0.997)

    def test_roundtrip_keeps_slopes_points_and_r2(self):
        from motorsports_data_notebook.suspension import AxleShockCalibration, ShockCalibration

        cal = ShockCalibration(
            front=AxleShockCalibration(
                1.1433,
                1.0406,
                damper_per_wheel_fit=0.9082,
                wheel_r2=0.9961,
                damper_r2=0.9985,
                damper_wheel_r2=0.9979,
                points=[[401.4, 300.0, 10.2], [407.6, 295.0, 15.1]],
            ),
            rear=AxleShockCalibration(1.3484, 1.0407, excluded_points=[1, 2]),
        )
        original = VehicleProfile(name="KK-F", shock_calibration=cal)
        restored = _profile_from_dict(_profile_to_dict(original))
        assert restored.shock_calibration is not None
        assert restored.shock_calibration.front.wheel_per_sensor == pytest.approx(1.1433)
        assert restored.shock_calibration.front.wheel_r2 == pytest.approx(0.9961)
        assert restored.shock_calibration.front.damper_per_wheel == pytest.approx(0.9082)
        assert restored.shock_calibration.front.damper_wheel_r2 == pytest.approx(0.9979)
        # no direct fit stored -> falls back to the ratio of slopes
        assert restored.shock_calibration.rear.damper_per_wheel == pytest.approx(1.0407 / 1.3484)
        assert restored.shock_calibration.front.points == [
            [401.4, 300.0, 10.2],
            [407.6, 295.0, 15.1],
        ]
        assert restored.shock_calibration.rear.excluded_points == [1, 2]
        assert restored.shock_calibration.rear.wheel_r2 is None


class TestBuiltinKKF:
    """The KK-F shares logger 8401203 with the KK-SII."""

    def test_kk_f_resolves_by_vehicle(self):
        profile = get_profile_for_logger("8401203", "KK-F")
        assert profile is not None
        assert profile.name == "KK-F"
        assert profile.channel_names["shock_fl"] == "LF_Shock_Pot"
        # analysis motion ratios are pot/wheel = 1 / wheel_per_sensor from the calibration
        assert profile.shock_calibration is not None
        assert profile.motion_ratios.front_left == pytest.approx(1 / 1.1433, rel=1e-3)
        assert profile.motion_ratios.rear_left == pytest.approx(1 / 1.3484, rel=1e-3)
        # damper/wheel is the directly fitted slope, not the ratio of sensor slopes (0.910)
        assert profile.shock_calibration.front.damper_per_wheel == pytest.approx(0.9082)
        assert profile.shock_calibration.rear.damper_per_wheel == pytest.approx(0.7713)
        assert profile.shock_calibration.front.damper_wheel_r2 == pytest.approx(0.9979)
        assert profile.shock_calibration.rear.excluded_points == [1, 2]

    def test_kk_sii_is_default_for_shared_logger(self):
        assert get_profile_for_logger("8401203").name == "KK-SII"
        assert get_profile_for_logger("8401203", "KK-SII").name == "KK-SII"
        assert (
            get_profile_for_logger("8401203", "KK-SII").channel_names["shock_rr"] == "RR_Shock_Pot"
        )

    def test_kk_f_from_xrz_file_name(self):
        log = MockLogFile(
            metadata={"Logger ID": 8401203, "Vehicle": ""},
            file_name="CMD_KK-F_Tsukuba_Car_Race_a_0344.xrz",
        )
        assert get_profile_for_log(log).name == "KK-F"

    def test_kk_f_from_xrk_metadata(self):
        log = MockLogFile(metadata={"Logger ID": 8401203, "Vehicle": "KK-F"}, file_name="x.xrk")
        assert get_profile_for_log(log).name == "KK-F"


class TestBuiltin86PotWheel:
    """The 86 profile uses the same pot/wheel convention via shock_calibration."""

    def test_86_ratios_unchanged_and_from_calibration(self):
        profile = get_profile_for_logger("6701209")
        assert profile is not None and profile.name == "Toyota 86 ZN6"
        assert profile.shock_calibration is not None
        assert profile.shock_calibration.front.damper_per_sensor == pytest.approx(1.0)
        assert profile.shock_calibration.front.damper_per_wheel == pytest.approx(0.997)
        assert profile.shock_calibration.rear.damper_per_wheel == pytest.approx(0.768)
        assert profile.motion_ratios.front_left == pytest.approx(0.997)
        assert profile.motion_ratios.rear_left == pytest.approx(0.768)
        assert profile.motion_ratios == profile.shock_calibration.pot_wheel_ratios()

    def test_every_builtin_profile_has_calibration(self):
        for key, profile in load_builtin_profiles().items():
            if key == "iracing" or key == "15074" or key == "6603435":
                continue  # no shock pots (GPS-only / velocity channels)
            assert profile.shock_calibration is not None, key


class TestTrackWidthsProfiles:
    """track_width_mm section round-trips and the builtin cars carry it."""

    def test_roundtrip(self):
        from motorsports_data_notebook.roll_gradient import TrackWidths

        original = VehicleProfile(name="Car", track_widths=TrackWidths(1520, 1540))
        restored = _profile_from_dict(_profile_to_dict(original))
        assert restored.track_widths is not None
        assert restored.track_widths.front_mm == pytest.approx(1520)
        assert restored.track_widths.rear_mm == pytest.approx(1540)

    def test_missing_or_invalid_is_none(self):
        assert _profile_from_dict({"name": "x"}).track_widths is None
        assert (
            _profile_from_dict(
                {"name": "x", "track_width_mm": {"front": 0, "rear": 1}}
            ).track_widths
            is None
        )
        assert (
            _profile_from_dict({"name": "x", "track_width_mm": {"front": 1500}}).track_widths
            is None
        )

    def test_builtin_values(self):
        t86 = get_profile_for_logger("6701209").track_widths
        assert (t86.front_mm, t86.rear_mm) == (1520, 1540)
        kkf = get_profile_for_logger("8401203", "KK-F")
        assert (kkf.track_widths.front_mm, kkf.track_widths.rear_mm) == (1498, 1503)
        # KK-F channel set from the Tsukuba race config (not the KK-SII names)
        assert kkf.channel_names["throttle"] == "Throttle_PCT"
        assert kkf.channel_names["brake"] == "P_Brake_Actual"
        assert kkf.channel_names["brake_rear"] == ""
        kks = get_profile_for_logger("8401203", "KK-SII")
        assert kks.channel_names["throttle"] == "Throttle"
        assert kks.channel_names["brake"] == "F_BRK"
