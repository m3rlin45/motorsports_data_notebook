"""The heat-frame switch (#376): which series every loader of the fit hands out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from motorsports_data_notebook.tire_model import heat_frame as hf


def _kin() -> pa.Table:
    return pa.table(
        {
            "t_session_s": [0.0, 1.0],
            "lap_num": [1, 1],
            "speed_ms": [30.0, 30.0],
            "lat_g": [0.5, 0.6],
            "long_g": [0.1, -0.2],
            "lat_source": ["imu", "imu"],
        }
    )


class TestUsing:
    def test_the_tyres_frame_is_the_default(self) -> None:
        assert hf.current() == hf.TYRES

    def test_sets_and_restores_the_frame(self) -> None:
        with hf.using(hf.SENSOR):
            assert hf.current() == hf.SENSOR
            with hf.using(hf.TYRES):
                assert hf.current() == hf.TYRES
            assert hf.current() == hf.SENSOR
        assert hf.current() == hf.TYRES

    def test_restores_after_an_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with hf.using(hf.SENSOR):
                raise RuntimeError("boom")
        assert hf.current() == hf.TYRES

    def test_none_is_the_frame_in_force(self) -> None:
        assert hf.resolve(None) == hf.TYRES
        with hf.using(hf.SENSOR):
            assert hf.resolve(None) == hf.SENSOR
            assert hf.resolve(hf.TYRES) == hf.TYRES
        with pytest.raises(ValueError):
            hf.resolve("ground")

    def test_the_committed_models_frame(self, tmp_path: Path) -> None:
        import json

        assert hf.model_frame(tmp_path) == hf.current()  # no model: the frame in force
        (tmp_path / "tire_model.json").write_text(json.dumps({"heat_input": {"form": "x"}}))
        assert hf.model_frame(tmp_path) == hf.SENSOR  # fitted before #376
        (tmp_path / "tire_model.json").write_text(json.dumps({"heat_input": {"frame": "tyres"}}))
        assert hf.model_frame(tmp_path) == hf.TYRES

    def test_rejects_an_unknown_frame(self) -> None:
        with pytest.raises(ValueError):
            with hf.using("ground"):
                pass

    def test_the_tracks_dir_is_restored(self, tmp_path: Path) -> None:
        with hf.using(hf.TYRES, tracks_dir=tmp_path):
            assert hf._tracks_dir() == tmp_path
        assert hf._STATE["tracks_dir"] is None


class TestFrameOf:
    def test_a_model_without_a_frame_is_sensor(self) -> None:
        assert hf.frame_of({}) == hf.SENSOR
        assert hf.frame_of({"heat_input": {"form": "..."}}) == hf.SENSOR

    def test_reads_the_recorded_frame(self) -> None:
        assert hf.frame_of({"heat_input": {"frame": "tyres"}}) == hf.TYRES

    def test_rejects_an_unknown_recorded_frame(self) -> None:
        with pytest.raises(ValueError):
            hf.frame_of({"heat_input": {"frame": "imu"}})


class TestLoadHeatKinematics:
    def test_sensor_is_load_kinematics_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import motorsports_data_notebook.tire_etl.kinematics as km

        kin = _kin()
        monkeypatch.setattr(km, "load_kinematics", lambda root, sid: kin)
        assert hf.load_heat_kinematics(Path("."), "s", hf.SENSOR) is kin
        with hf.using(hf.SENSOR):
            assert hf.load_heat_kinematics(Path("."), "s") is kin

    def test_tyres_rebuilds_from_the_sessions_layout_and_track(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import motorsports_data_notebook.tire_etl.kinematics as km
        from motorsports_data_notebook.tire_model.lapsim import tyres_frame as tf

        kin = _kin()
        seen: dict = {}
        monkeypatch.setattr(km, "load_kinematics", lambda root, sid: kin)
        monkeypatch.setattr(hf, "session_layout", lambda root, sid: "tsukuba_2000")
        monkeypatch.setattr(hf, "layout_track", lambda layout: f"track:{layout}")
        monkeypatch.setattr(hf, "_GROUND", {})

        def fake(root, sid, table, layout, track):  # type: ignore[no-untyped-def]
            seen.update(sid=sid, layout=layout, track=track, table=table)
            return "rebuilt"

        monkeypatch.setattr(tf, "tyres_kinematics", fake)
        got = hf.load_heat_kinematics(Path("."), "s")  # the default frame
        assert got == "rebuilt"
        assert seen == {
            "sid": "s",
            "layout": "tsukuba_2000",
            "track": "track:tsukuba_2000",
            "table": kin,
        }
        # an explicit frame wins over the one in force
        with hf.using(hf.TYRES):
            assert hf.load_heat_kinematics(Path("."), "s", hf.SENSOR) is kin


class TestFallbacks:
    """A session the tyres' frame cannot be built for is said so, never crashes."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import motorsports_data_notebook.tire_etl.kinematics as km

        monkeypatch.setattr(hf, "_GROUND", {})
        monkeypatch.setattr(hf, "_LEFT_OUT", {})
        monkeypatch.setattr(km, "load_kinematics", lambda root, sid: _kin())

    def test_no_layout_is_left_out_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(hf, "session_layout", lambda root, sid: None)
        with caplog.at_level("WARNING"):
            assert hf.load_heat_kinematics(Path("."), "s") is None
        assert "s has no heat in this frame" in caplog.text
        assert hf.fallbacks()["left_out_sessions"] == {"s": "no layout in the sessions table"}

    def test_a_layout_unknown_to_the_geometry_is_left_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hf, "session_layout", lambda root, sid: "new_circuit")
        assert hf.load_heat_kinematics(Path("."), "s") is None
        why = hf.fallbacks()["left_out_sessions"]["s"]
        assert "new_circuit" in why and "geometry" in why

    def test_no_committed_track_is_the_ground_frame_and_said_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from motorsports_data_notebook.tire_model.lapsim import tyres_frame as tf

        monkeypatch.setattr(hf, "session_layout", lambda root, sid: "tsukuba_2000")
        monkeypatch.setattr(hf, "_TRACKS", {})
        seen: dict = {}

        def fake(root, sid, table, layout, track):  # type: ignore[no-untyped-def]
            seen["track"] = track
            return table

        monkeypatch.setattr(tf, "tyres_kinematics", fake)
        with caplog.at_level("WARNING"), hf.using(hf.TYRES, tracks_dir=tmp_path):
            assert hf.load_heat_kinematics(Path("."), "s") is not None
            assert hf.load_heat_kinematics(Path("."), "t") is not None
        assert seen["track"] is None  # the ground frame
        assert caplog.text.count("no committed track for layout tsukuba_2000") == 1
        got = hf.fallbacks()
        assert got["ground_frame_layouts"] == {"tsukuba_2000": ["s", "t"]}
        assert hf.fallbacks({"t"})["ground_frame_layouts"] == {"tsukuba_2000": ["t"]}
        assert list(tmp_path.iterdir()) == []  # never built

    def test_a_root_without_a_sessions_table_is_an_error_not_a_skip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from motorsports_data_notebook.tire_model import pace_sim

        monkeypatch.setattr(hf, "_LAYOUTS", {})
        with pytest.raises(hf.HeatFrameError, match="sensor"):
            hf.load_heat_kinematics(tmp_path, "s")
        # the pace model's per-file skip does not swallow it
        (tmp_path / "timeseries" / "2026-01").mkdir(parents=True)
        (tmp_path / "timeseries" / "2026-01" / "s.parquet").write_bytes(b"")
        monkeypatch.setattr(pace_sim, "_SAMPLE_CACHE", {})
        with pytest.raises(hf.HeatFrameError):
            pace_sim.load_lap_samples(tmp_path)

    def test_no_gps_geometry_is_left_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from motorsports_data_notebook.tire_model.lapsim import tyres_frame as tf

        monkeypatch.setattr(hf, "session_layout", lambda root, sid: "tsukuba_2000")
        monkeypatch.setattr(hf, "layout_track", lambda layout: None)
        monkeypatch.setattr(tf, "tyres_kinematics", lambda *a: None)
        assert hf.load_heat_kinematics(Path("."), "s") is None
        assert hf.fallbacks()["left_out_sessions"] == {"s": "no GPS geometry"}
        assert hf.fallbacks()["ground_frame_layouts"] == {}


class TestFitAndScoreFollowTheFrame:
    def test_build_warmup_table_fits_in_the_frame_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from motorsports_data_notebook.tire_model import warmup_table as wt

        monkeypatch.setattr(wt, "_build_warmup_table", lambda *a, **k: {"frame": hf.current()})
        assert wt.build_warmup_table(Path("."), write_artifacts=False) == {"frame": hf.TYRES}
        got = wt.build_warmup_table(Path("."), write_artifacts=False, heat_frame=hf.SENSOR)
        assert got == {"frame": hf.SENSOR}
        assert hf.current() == hf.TYRES

    def test_fit_fold_passes_the_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from motorsports_data_notebook.tire_model import validate as va

        seen: dict = {}

        def fake(root, **kw):  # type: ignore[no-untyped-def]
            seen.update(kw)
            return {}

        monkeypatch.setattr(va, "build_warmup_table", fake)
        va._fit_fold(Path("."), ["a"], heat_frame=hf.SENSOR)
        assert seen["heat_frame"] == hf.SENSOR and seen["exclude_session_ids"] == {"a"}
        va._fit_fold(Path("."), ["a"])
        assert seen["heat_frame"] == hf.TYRES

    def test_session_series_are_cached_per_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import motorsports_data_notebook.tire_etl.dataset as ds
        from motorsports_data_notebook.tire_model import statespace as ss

        calls: list[str] = []

        def fake(root, sid, frame=None):  # type: ignore[no-untyped-def]
            calls.append(str(frame))
            return _kin()

        monkeypatch.setattr(hf, "load_heat_kinematics", fake)
        monkeypatch.setattr(ds, "load_group", lambda *a, **k: None)
        ss._session_timeseries_in.cache_clear()
        try:
            a = ss._session_timeseries("/nowhere", "s")
            b = ss._session_timeseries("/nowhere", "s")
            with hf.using(hf.SENSOR):
                c = ss._session_timeseries("/nowhere", "s")
        finally:
            ss._session_timeseries_in.cache_clear()
        assert calls == [hf.TYRES, hf.SENSOR]
        assert a is b and c is not a
        assert np.allclose(a.kin["lat_g"], [0.5, 0.6])


class TestServedFactor:
    def test_unit_factors_cover_every_kind_of_source(self) -> None:
        import pandas as pd

        from motorsports_data_notebook.tire_model.lapsim.pressure_gate import (
            heats_in_tyres_frame,
            unit_factors,
        )

        got = unit_factors(pd.DataFrame({"source": ["imu", "tyres"]}))
        assert set(got) == {"", "imu", "gps", "mixed", "tyres"}
        assert all(np.array_equal(v, np.ones(4)) for v in got.values())
        assert heats_in_tyres_frame()
        with hf.using(hf.SENSOR):
            assert not heats_in_tyres_frame()

    def test_one_in_the_tyres_frame_whatever_the_source(self) -> None:
        from motorsports_data_notebook.tire_model.lapsim.pressure_gate import served_factor

        factors = {"imu": np.array([0.9, 0.9, 0.9, 0.9]), "gps": np.array([1.1, 1.1, 1.1, 1.1])}
        tyres = {"heat_input": {"frame": "tyres"}}
        for source in ("imu", "gps", "", "mixed"):
            assert np.array_equal(served_factor(tyres, factors, source), np.ones(4))

    def test_the_cells_factor_by_source_in_the_sensor_frame(self) -> None:
        from motorsports_data_notebook.tire_model.lapsim.pressure_gate import served_factor

        factors = {"imu": np.array([0.9, 0.9, 0.9, 0.9])}
        got = served_factor({}, factors, "imu")
        assert got is not None and np.array_equal(got, factors["imu"])
        assert served_factor({}, factors, "gps") is None


class TestFoldCache:
    def test_a_tyres_fold_has_its_own_key_and_the_sensor_key_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import hashlib

        from motorsports_data_notebook.tire_model import validate as va
        from motorsports_data_notebook.tire_model.lapsim import pressure_gate as pg

        monkeypatch.setattr(pg, "_fit_signature", lambda root: "SIG")
        monkeypatch.setattr(pg, "_tyres_signature", lambda: "TYRES")
        fits: list[str] = []

        def fake(root, ids, heat_frame="tyres"):  # type: ignore[no-untyped-def]
            fits.append(heat_frame)
            return {"heat_input": {"frame": heat_frame}} if heat_frame != "sensor" else {}

        monkeypatch.setattr(va, "_fit_fold", fake)
        a = pg.fold_model(Path("."), ["x", "y"], tmp_path, heat_frame="sensor")
        b = pg.fold_model(Path("."), ["x", "y"], tmp_path)  # the default: tyres
        pg.fold_model(Path("."), ["y", "x"], tmp_path, heat_frame="sensor")  # cached
        pg.fold_model(Path("."), ["x", "y"], tmp_path, heat_frame="tyres")  # cached
        assert fits == ["sensor", "tyres"]
        assert a == {} and b == {"heat_input": {"frame": "tyres"}}
        sensor_key = hashlib.sha256(("x\ny" + "SIG").encode()).hexdigest()[:24]
        assert (tmp_path / "pressure_folds" / f"{sensor_key}.json").exists()
        assert len(list((tmp_path / "pressure_folds").glob("*.json"))) == 2
