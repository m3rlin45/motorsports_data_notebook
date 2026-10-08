using System.IO;
using System.Reflection;
using TirePressureCalculator.Services.Modeling;

namespace TirePressureCalculator.Tests;

public class TireModelTests
{
    private static TireModel LoadBundled() =>
        TireModelLoader.LoadEmbedded(typeof(TireModel).Assembly);

    // ---------- Loader / schema-version gate ----------

    [Fact]
    public void LoadEmbedded_ParsesBundledArtifact()
    {
        var model = LoadBundled();
        Assert.InRange(model.Dto.SchemaVersion,
            TireModel.MinSupportedSchemaVersion, TireModel.SupportedSchemaVersion);
        Assert.NotNull(model.Dto.FitAtUtc);
        Assert.True(model.Dto.KBuckets.Count > 0, "expected at least one K bucket");
    }

    // Minimal valid-shaped artifact JSON at a given schema_version; bypasses
    // the embedded path so the tests can feed a stream directly.
    private static string MinimalArtifactJson(int schemaVersion) => $$$"""
        {
          "schema_version": {{{schemaVersion}}},
          "fit_at_utc": "2026-01-01T00:00:00Z",
          "model_form": "old",
          "gay_lussac": {"p_atm_bar": 1.0, "t_zero_c_to_k": 273.15, "t_cold_uses": "T_air"},
          "energy_balance": {"w_road": 0.2, "w_road_fitted": false,
            "t_road_proxy": {"formula": "x", "delta_sun_max_c": 10.0, "sun_factor_default": 1.0}},
          "conditions": {"values": ["dry"], "default": "dry"},
          "corners": ["fl", "fr", "rl", "rr"],
          "min_samples_per_bucket": 5,
          "priors_when_no_fit": {"tau_sec_seconds": 240.0, "K_kelvin_per_g2": 60.0},
          "tau_sec_by_car_corner_cond": [],
          "K_buckets": [],
          "g2_typ_by_track_car_cond": [],
          "lap_time_typ_by_track_car_cond": []
        }
        """;

    private static TireModel LoadFromJson(string json)
    {
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(json));
        return TireModelLoader.LoadFromStream(stream);
    }

    // v4 and below carried K values fitted against the per-track c_track,
    // which v5 dropped; they are no longer loadable.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(6)]
    public void Loader_RefusesUnsupportedSchemaVersion(int schemaVersion)
    {
        Assert.Throws<InvalidOperationException>(() => LoadFromJson(MinimalArtifactJson(schemaVersion)));
    }

    [Fact]
    public void Loader_AcceptsSchemaVersion5()
    {
        var model = LoadFromJson(MinimalArtifactJson(5));
        Assert.Equal(5, model.Dto.SchemaVersion);
    }

    [Fact]
    public void Loader_IgnoresLegacyCTrackKeys()
    {
        // A v5 artifact written by an older trainer may still carry the
        // c_track keys (all 1.0); they are unknown to the DTOs and ignored.
        var json = MinimalArtifactJson(5)
            .Replace("\"K_kelvin_per_g2\": 60.0}", "\"K_kelvin_per_g2\": 60.0, \"c_track\": 1.0}")
            .Replace("\"K_buckets\": [],",
                "\"K_buckets\": [],\n  \"c_track_by_track\": [{\"track_canonical\": \"synth\", \"value\": 1.0, \"stderr\": 0.0, \"n_buckets_used\": 4, \"anchor\": true}],");
        Assert.Contains("c_track_by_track", json);
        var model = LoadFromJson(json);
        Assert.Equal(60.0, model.Dto.PriorsWhenNoFit.KKelvinPerG2);
    }

    [Fact]
    public void Loader_ParsesV5PerCornerFieldsAndIgnoresHeatInputBlock()
    {
        const string json = """
            {
              "schema_version": 5,
              "fit_at_utc": "2026-01-01T00:00:00Z",
              "model_form": "v5",
              "gay_lussac": {"p_atm_bar": 1.0, "t_zero_c_to_k": 273.15, "t_cold_uses": "T_air"},
              "energy_balance": {"w_road": 0.2, "w_road_fitted": false,
                "t_road_proxy": {"formula": "x", "delta_sun_max_c": 10.0, "sun_factor_default": 1.0}},
              "conditions": {"values": ["dry"], "default": "dry"},
              "corners": ["fl", "fr", "rl", "rr"],
              "min_samples_per_bucket": 5,
              "priors_when_no_fit": {"tau_sec_seconds": 240.0, "K_kelvin_per_g2": 60.0},
              "tau_sec_by_car_corner_cond": [],
              "K_buckets": [],
              "g2_typ_by_track_car_cond": [
                {"track_canonical": "synth", "car": "Car", "condition": "dry", "g2_typ": 1.0, "n_laps_used": 10,
                 "q_typ_by_corner": {"fl": 1.3, "fr": 0.9, "rl": 1.1, "rr": 0.7}},
                {"track_canonical": "other", "car": "Car", "condition": "dry", "g2_typ": 0.8, "n_laps_used": 3}
              ],
              "lap_time_typ_by_track_car_cond": [],
              "outlap_typ_by_track_car_cond": [
                {"track_canonical": "synth", "car": "Car", "condition": "dry",
                 "outlap_moving_s": 90.0, "outlap_g2": 0.4, "n_laps_used": 10,
                 "outlap_q_by_corner": {"fl": 0.6, "fr": 0.3, "rl": 0.5, "rr": 0.2}}
              ],
              "heat_input": {"form": "q = ...", "v_ref_ms": 30.0,
                "geometry_by_car": {"Car": {"wheelbase_m": 2.5}},
                "drive_by_car": {"Car": "rwd"}, "brake_by_car": {"Car": {"front_bias": 0.6}}}
            }
            """;
        var model = LoadFromJson(json);

        var entry = model.Dto.G2TypByTrackCarCond.Single(r => r.TrackCanonical == "synth");
        Assert.NotNull(entry.QTypByCorner);
        Assert.Equal(1.3, entry.QTypByCorner!["fl"]);
        Assert.Null(model.Dto.G2TypByTrackCarCond.Single(r => r.TrackCanonical == "other").QTypByCorner);

        // Corner-aware lookup resolves the per-corner value; no corner (or no
        // map) keeps the mean so pre-v5 numbers are preserved.
        Assert.Equal(1.1, model.LookupG2("synth", "Car", "dry", "rl").Value);
        Assert.Equal(1.0, model.LookupG2("synth", "Car", "dry").Value);
        Assert.Equal(0.8, model.LookupG2("other", "Car", "dry", "rl").Value);
        // Pooled fallbacks (condition chain exhausted) average the per-corner value.
        var wet = model.LookupG2("synth", "Car", "wet", "fr");
        Assert.Equal(0.9, wet.Value);
        Assert.Equal("fallback(dry)", wet.Source);

        var outlap = model.LookupOutlap("synth", "Car", "dry", "rr");
        Assert.NotNull(outlap);
        Assert.Equal(0.2, outlap!.Value.G2);
        Assert.Equal(90.0, outlap.Value.MovingSeconds);
        Assert.Equal(0.4, model.LookupOutlap("synth", "Car", "dry")!.Value.G2);
    }

    [Fact]
    public void LookupG2AndOutlap_CornerArgumentIsInertWithoutPerCornerFields()
    {
        // The committed artifact may or may not carry the v5 fields; when it
        // does not, passing a corner must not change a single number.
        var model = LoadBundled();
        if (model.Dto.G2TypByTrackCarCond.Any(r => r.QTypByCorner is not null)) return;
        foreach (var track in model.AvailableTracks)
            foreach (var car in model.AvailableCars)
                foreach (var corner in new[] { "fl", "fr", "rl", "rr" })
                {
                    Assert.Equal(model.LookupG2(track, car, "dry").Value,
                        model.LookupG2(track, car, "dry", corner).Value);
                    Assert.Equal(model.LookupOutlap(track, car, "dry"),
                        model.LookupOutlap(track, car, "dry", corner));
                }
    }

    // ---------- Available cars / tracks / conditions ----------

    [Fact]
    public void AvailableCars_IncludesBothExpectedCars()
    {
        var model = LoadBundled();
        Assert.Contains("Inferno 86", model.AvailableCars);
        Assert.Contains("FJ", model.AvailableCars);
    }

    [Fact]
    public void ResolveCar_MapsAliasedNamesToPooledLabel()
    {
        var model = LoadBundled();
        Assert.Equal("FJ", model.ResolveCar("KK-F"));
        Assert.Equal("FJ", model.ResolveCar("KK-SII"));
        Assert.Equal("FJ", model.ResolveCar("FJ"));
        Assert.Equal("Inferno 86", model.ResolveCar("Inferno 86"));
    }

    [Fact]
    public void AvailableTracks_ListsEveryTrackWithATypicalHeatInput()
    {
        var model = LoadBundled();
        Assert.Contains("tsukuba_2000", model.AvailableTracks);
        var observed = model.Dto.G2TypByTrackCarCond.Select(r => r.TrackCanonical).Distinct().OrderBy(s => s);
        Assert.Equal(observed, model.AvailableTracks);
    }

    [Fact]
    public void AvailableConditions_ContainsDryDampWet()
    {
        var model = LoadBundled();
        Assert.Equal(new[] { "dry", "damp", "wet" }, model.AvailableConditions);
        Assert.Equal("dry", model.DefaultCondition);
    }

    [Fact]
    public void GayLussac_TColdUses_MatchesPythonConvention()
    {
        // Pinning this prevents accidental schema drift: the manual calculator
        // and the predictor must agree that the Gay-Lussac cold side uses T_air.
        var model = LoadBundled();
        Assert.Equal("T_air", model.Dto.GayLussac.TColdUses);
    }

    // ---------- Condition fallback chain ----------

    [Fact]
    public void ConditionChain_Dry_ReturnsDryOnly()
    {
        Assert.Equal(new[] { "dry" }, TireModel.ConditionChain("dry"));
    }

    [Fact]
    public void ConditionChain_Damp_FallsBackToDry()
    {
        Assert.Equal(new[] { "damp", "dry" }, TireModel.ConditionChain("damp"));
    }

    [Fact]
    public void ConditionChain_Wet_PrefersDampOverDry()
    {
        Assert.Equal(new[] { "wet", "damp", "dry" }, TireModel.ConditionChain("wet"));
    }

    // ---------- Lookup helpers ----------

    [Fact]
    public void LookupK_FJ_FL_Dry_HitsExactBucket()
    {
        var model = LoadBundled();
        var k = model.LookupK("FJ", "fl", "dry");
        Assert.False(k.FromPrior);
        Assert.Contains("FJ, fl, dry", k.SourceBucket);
        Assert.True(k.ValueKelvinPerG2 > 0);
    }

    [Fact]
    public void LookupK_FJ_Wet_ResolvesToAFittedBucketThroughTheChain()
    {
        // Rain τ/K are fitted per condition when a (car, track) rain bucket has
        // enough sessions, else the lookup walks wet → damp → dry. Either way a
        // wet FJ lookup must land on a fitted (FJ, fl, *) bucket, never a prior.
        var model = LoadBundled();
        var k = model.LookupK("FJ", "fl", "wet");
        Assert.False(k.FromPrior);
        Assert.StartsWith("(FJ, fl,", k.SourceBucket);
    }

    [Fact]
    public void LookupK_UnknownCar_ReturnsPrior()
    {
        var model = LoadBundled();
        var k = model.LookupK("Fictional Car", "fl", "dry");
        Assert.True(k.FromPrior);
        Assert.Equal(model.Dto.PriorsWhenNoFit.KKelvinPerG2, k.ValueKelvinPerG2);
    }

    [Fact]
    public void LookupTau_FJ_FL_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var tau = model.LookupTau("FJ", "fl", "dry");
        Assert.False(tau.FromPrior);
        // Plan says racing-tire τ is typically 150–350 s.
        // τ is measured from pit exit with the stint anchored on the pit-exit
        // TPMS reading (v0.26): 400–800 s on the fleet, longer than the v0
        // numbers that anchored after the out-lap.
        Assert.InRange(tau.ValueSeconds, 100.0, 900.0);
    }

    [Fact]
    public void LookupG2_Tsukuba_FJ_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var g2 = model.LookupG2("tsukuba_2000", "FJ", "dry");
        // FJ at Tsukuba on dry runs ~1.0 G² per plan
        Assert.InRange(g2.Value, 0.5, 1.5);
        Assert.Equal("exact", g2.Source);
    }

    [Fact]
    public void LookupG2_NewTrack_FallsBackToGlobalMean()
    {
        var model = LoadBundled();
        var g2 = model.LookupG2("zandvoort", "FJ", "dry");
        // No data for zandvoort at all → no track-car, no track-only,
        // chain falls all the way to global.
        Assert.Equal("global", g2.Source);
        Assert.InRange(g2.Value, 0.3, 1.5);
    }

    [Fact]
    public void LookupLapTime_Tsukuba_FJ_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var lt = model.LookupLapTime("tsukuba_2000", "FJ", "dry");
        // Tsukuba FJ typical lap ~ 60-70 s
        Assert.InRange(lt.ValueSeconds, 40.0, 90.0);
        Assert.Equal("exact", lt.Source);
    }
}
