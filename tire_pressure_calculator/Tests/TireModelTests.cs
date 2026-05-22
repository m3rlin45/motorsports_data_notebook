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
        Assert.Equal(TireModel.SupportedSchemaVersion, model.Dto.SchemaVersion);
        Assert.NotNull(model.Dto.FitAtUtc);
        Assert.True(model.Dto.KBuckets.Count > 0, "expected at least one K bucket");
    }

    [Fact]
    public void Loader_RefusesUnsupportedSchemaVersion()
    {
        // Craft a minimal valid-shaped JSON with schema_version=1 and ensure the
        // wrapper throws. We bypass the embedded path and feed a stream directly.
        const string oldSchema = """
            {
              "schema_version": 1,
              "fit_at_utc": "2026-01-01T00:00:00Z",
              "model_form": "old",
              "gay_lussac": {"p_atm_bar": 1.0, "t_zero_c_to_k": 273.15, "t_cold_uses": "T_air"},
              "energy_balance": {"w_road": 0.2, "w_road_fitted": false,
                "t_road_proxy": {"formula": "x", "delta_sun_max_c": 10.0, "sun_factor_default": 1.0}},
              "conditions": {"values": ["dry"], "default": "dry"},
              "corners": ["fl", "fr", "rl", "rr"],
              "min_samples_per_bucket": 5,
              "priors_when_no_fit": {"tau_sec_seconds": 240.0, "K_kelvin_per_g2": 60.0, "c_track": 1.0},
              "tau_sec_by_car_corner_cond": [],
              "K_buckets": [],
              "c_track_by_track": [],
              "g2_typ_by_track_car_cond": [],
              "lap_time_typ_by_track_car_cond": []
            }
            """;
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(oldSchema));
        Assert.Throws<InvalidOperationException>(() => TireModelLoader.LoadFromStream(stream));
    }

    // ---------- Available cars / tracks / conditions ----------

    [Fact]
    public void AvailableCars_IncludesBothExpectedCars()
    {
        var model = LoadBundled();
        Assert.Contains("Inferno 86", model.AvailableCars);
        Assert.Contains("KK-SII", model.AvailableCars);
    }

    [Fact]
    public void AvailableTracks_IncludesTsukubaAsAnchor()
    {
        var model = LoadBundled();
        Assert.Contains("tsukuba_2000", model.AvailableTracks);
        var anchor = model.Dto.CTrackByTrack.Single(r => r.TrackCanonical == "tsukuba_2000");
        Assert.True(anchor.Anchor, "tsukuba_2000 must be the c_track anchor (value == 1.0)");
        Assert.Equal(1.0, anchor.Value, precision: 9);
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
    public void LookupK_KKSII_FL_Dry_HitsExactBucket()
    {
        var model = LoadBundled();
        var k = model.LookupK("KK-SII", "fl", "dry");
        Assert.False(k.FromPrior);
        Assert.Contains("KK-SII, fl, dry", k.SourceBucket);
        Assert.True(k.ValueKelvinPerG2 > 0);
    }

    [Fact]
    public void LookupK_KKSII_Wet_FallsBackToDampOrDry()
    {
        // KK-SII has no wet K bucket in the current artifact (Fuji wet samples
        // didn't meet the τ-fit threshold). Wet should fall through the chain
        // and produce a (KK-SII, *, damp) or (KK-SII, *, dry) source bucket.
        var model = LoadBundled();
        var k = model.LookupK("KK-SII", "fl", "wet");
        Assert.False(k.FromPrior);
        Assert.True(k.SourceBucket.Contains("damp") || k.SourceBucket.Contains("dry"),
            $"Expected damp/dry fallback for KK-SII wet; got {k.SourceBucket}");
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
    public void LookupTau_KKSII_FL_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var tau = model.LookupTau("KK-SII", "fl", "dry");
        Assert.False(tau.FromPrior);
        // Plan says racing-tire τ is typically 150–350 s.
        Assert.InRange(tau.ValueSeconds, 100.0, 500.0);
    }

    [Fact]
    public void LookupCTrack_Tsukuba_IsExactlyOne()
    {
        var model = LoadBundled();
        var c = model.LookupCTrack("tsukuba_2000");
        Assert.Equal(1.0, c.Value, precision: 9);
        Assert.False(c.FromPrior);
    }

    [Fact]
    public void LookupCTrack_UnknownTrack_ReturnsPrior()
    {
        var model = LoadBundled();
        var c = model.LookupCTrack("zandvoort");
        Assert.True(c.FromPrior);
        Assert.Equal(model.Dto.PriorsWhenNoFit.CTrack, c.Value);
    }

    [Fact]
    public void LookupG2_Tsukuba_KKSII_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var g2 = model.LookupG2("tsukuba_2000", "KK-SII", "dry");
        // KK-SII at Tsukuba on dry runs ~1.0 G² per plan
        Assert.InRange(g2.Value, 0.5, 1.5);
        Assert.Equal("exact", g2.Source);
    }

    [Fact]
    public void LookupG2_NewTrack_FallsBackToGlobalMean()
    {
        var model = LoadBundled();
        var g2 = model.LookupG2("zandvoort", "KK-SII", "dry");
        // No data for zandvoort at all → no track-car, no track-only,
        // chain falls all the way to global.
        Assert.Equal("global", g2.Source);
        Assert.InRange(g2.Value, 0.3, 1.5);
    }

    [Fact]
    public void LookupLapTime_Tsukuba_KKSII_Dry_IsInPhysicalRange()
    {
        var model = LoadBundled();
        var lt = model.LookupLapTime("tsukuba_2000", "KK-SII", "dry");
        // Tsukuba KK-SII typical lap ~ 60-70 s
        Assert.InRange(lt.ValueSeconds, 40.0, 90.0);
        Assert.Equal("exact", lt.Source);
    }
}
