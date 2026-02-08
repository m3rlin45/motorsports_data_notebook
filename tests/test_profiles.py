"""Tests for profiles module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest
import yaml

from motorsports_data_notebook.profiles import (
    ALL_CANONICAL_KEYS,
    DEFAULT_CHANNEL_NAMES,
    VehicleProfile,
    _get_user_profiles_path,
    _load_profiles_file,
    _profile_from_dict,
    _profile_to_dict,
    _validate_channel_names,
    get_logger_id,
    get_profile_for_logger,
    load_builtin_profiles,
    load_user_profiles,
    save_profile_for_logger,
    save_user_profiles,
)
from motorsports_data_notebook.suspension import MotionRatios


@dataclass
class MockLogFile:
    """Mock LogFile for testing."""

    metadata: Dict[str, str] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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
    """Tests for _validate_channel_names."""

    def test_valid_keys(self):
        """Should not raise when all required keys are present."""
        _validate_channel_names(
            {"throttle": "PPS", "brake": "BP"}, ["throttle", "brake"], "test_func"
        )

    def test_missing_keys(self):
        """Should raise KeyError listing missing keys."""
        with pytest.raises(KeyError, match="Missing.*brake"):
            _validate_channel_names({"throttle": "PPS"}, ["throttle", "brake"], "test_func")


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
        builtin_profile = VehicleProfile(name="Builtin Car", channel_names={"throttle": "BuiltinTPS"})
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
        """Builtin profiles YAML should load without error."""
        result = load_builtin_profiles()
        # The builtin file has no logger_map entries, so result should be empty
        assert isinstance(result, dict)

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

        # Each profile should have all canonical keys
        for profile_id, profile_data in data["profiles"].items():
            assert "name" in profile_data
            assert "channel_names" in profile_data
            for key in ALL_CANONICAL_KEYS:
                assert key in profile_data["channel_names"], (
                    f"Profile '{profile_id}' missing canonical key '{key}'"
                )


import importlib.resources
