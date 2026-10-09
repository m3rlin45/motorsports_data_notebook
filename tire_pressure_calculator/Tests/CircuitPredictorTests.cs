using System.IO;
using System.Text.Json;
using TirePressureCalculator.Services;
using TirePressureCalculator.Services.Modeling;

namespace TirePressureCalculator.Tests;

public class CircuitPredictorTests
{
    private static readonly string[] Corners = { "fl", "fr", "rl", "rr" };

    private static CircuitPredictor LoadPredictor() =>
        new(TireModelLoader.LoadEmbedded(typeof(TireModel).Assembly));

    // ---------- Python parity: same inputs, same per-corner cold pressure ----------

    [Fact]
    public void Predict_MatchesPythonOutputOnVendoredFixture()
    {
        var fixturePath = Path.Combine(
            Path.GetDirectoryName(typeof(CircuitPredictorTests).Assembly.Location)!,
            "Fixtures", "python_predictions.json");
        Assert.True(File.Exists(fixturePath),
            $"Fixture not found at {fixturePath}. Re-generate with the script in CircuitPredictorTests.");

        using var stream = File.OpenRead(fixturePath);
        var cases = JsonSerializer.Deserialize<List<FixtureCase>>(stream)
            ?? throw new InvalidDataException("Fixture deserialized to null");
        Assert.NotEmpty(cases);

        var predictor = LoadPredictor();
        foreach (var c in cases)
        {
            foreach (var corner in Corners)
            {
                var prediction = predictor.Predict(
                    track: c.Inputs.Track,
                    car: c.Inputs.Car,
                    condition: c.Inputs.TrackCondition,
                    lapWithinStint: c.Inputs.LapWithinStint,
                    ambientTempC: c.Inputs.AmbientTempC,
                    trackTempC: c.Inputs.TrackTempC,
                    cloudCoverPct: c.Inputs.CloudCoverPct,
                    corner: corner,
                    targetHotPressureBar: c.Inputs.TargetHotPressureBar,
                    coldTireTempC: c.Inputs.ColdTireTempC,
                    targetLapTimeS: c.Inputs.TargetLapTimeS,
                    compound: c.Inputs.Compound);
                var py = c.Corners[corner];
                Assert.True(
                    Math.Abs(prediction.ColdPressureBar - py.ColdPressureBar) < 1e-3,
                    $"[{c.Label}/{corner}] cold pressure C#={prediction.ColdPressureBar:F6} " +
                    $"vs Python={py.ColdPressureBar:F6}");
                Assert.True(
                    Math.Abs(prediction.PredictedHotTempC - py.PredictedHotTempC) < 1e-2,
                    $"[{c.Label}/{corner}] predicted hot temp C#={prediction.PredictedHotTempC:F4} " +
                    $"vs Python={py.PredictedHotTempC:F4}");
                Assert.Equal(string.Join(", ", py.KSourceBucket),
                    StripParens(prediction.KSourceBucket));
                if (c.G2Scale is double expectedScale)
                {
                    Assert.True(Math.Abs(prediction.G2Scale - expectedScale) < 1e-9,
                        $"[{c.Label}/{corner}] g2 scale C#={prediction.G2Scale:F6} " +
                        $"vs Python={expectedScale:F6}");
                    Assert.Equal(c.G2PaceSource, prediction.G2PaceSource);
                }
            }
        }
    }

    private static string StripParens(string bucket) =>
        bucket.TrimStart('(').TrimEnd(')');

    // ---------- Targeted unit tests mirroring predict.py tests ----------

    [Fact]
    public void Predict_Wet_ResolvesThroughTheConditionChain()
    {
        var p = LoadPredictor();
        var result = p.Predict(
            track: "tsukuba_2000",
            car: "KK-SII",
            condition: "wet",
            lapWithinStint: 5,
            ambientTempC: 15.0,
            trackTempC: null,
            cloudCoverPct: 100.0,
            corner: "fl",
            targetHotPressureBar: 1.7);
        // Rain buckets are fitted on their own when they have enough sessions
        // and fall back through wet → damp → dry otherwise; either way the
        // lookup must land on a fitted (FJ, fl, *) bucket, never a prior.
        Assert.Contains("(FJ, fl,", result.KSourceBucket);
        Assert.False(result.KFromPrior);
    }

    [Fact]
    public void Predict_ColdTireTempOverride_IsTheWarmupStartAndGayLussacColdSide()
    {
        var p = LoadPredictor();
        var noOverride = p.Predict(
            track: "tsukuba_2000", car: "KK-SII", condition: "dry",
            lapWithinStint: 5, ambientTempC: 15.0,
            trackTempC: null, cloudCoverPct: 100.0,
            corner: "fl", targetHotPressureBar: 1.7);
        var warmTire = p.Predict(
            track: "tsukuba_2000", car: "KK-SII", condition: "dry",
            lapWithinStint: 5, ambientTempC: 15.0,
            trackTempC: null, cloudCoverPct: 100.0,
            corner: "fl", targetHotPressureBar: 1.7,
            coldTireTempC: 25.0);

        // T_eff is an ambient property and must not change.
        Assert.Equal(noOverride.TEffC, warmTire.TEffC, precision: 9);
        // The current tire temp is both the cold side and the warmup start.
        Assert.Equal(15.0, noOverride.TColdC, precision: 9);
        Assert.Equal(15.0, noOverride.TStartC, precision: 9);
        Assert.Equal(25.0, warmTire.TColdC, precision: 9);
        Assert.Equal(25.0, warmTire.TStartC, precision: 9);
        // A warmer start ends the lap hotter by the decayed start excess; the
        // decay runs over the out-lap segment plus the flying laps.
        double decay = Math.Exp(-(noOverride.OutlapTimeS + noOverride.TAtLapNs) / noOverride.TauSec);
        Assert.Equal(10.0 * decay, warmTire.PredictedHotTempC - noOverride.PredictedHotTempC, precision: 9);
        // Higher T_cold → higher recommended cold pressure (P/T const).
        Assert.True(warmTire.ColdPressureBar > noOverride.ColdPressureBar);
    }

    [Fact]
    public void Predict_RejectsUnknownCondition()
    {
        var p = LoadPredictor();
        Assert.Throws<ArgumentException>(() => p.Predict(
            track: "tsukuba_2000", car: "KK-SII", condition: "monsoon",
            lapWithinStint: 5, ambientTempC: 15.0,
            trackTempC: null, cloudCoverPct: null,
            corner: "fl", targetHotPressureBar: 1.7));
    }

    [Fact]
    public void Predict_CrossTrack_UsesSameKAndTauDifferentPerCornerQ()
    {
        var p = LoadPredictor();
        var tsk = p.Predict(
            track: "tsukuba_2000", car: "KK-SII", condition: "dry",
            lapWithinStint: 5, ambientTempC: 18.0,
            trackTempC: null, cloudCoverPct: 50.0,
            corner: "fl", targetHotPressureBar: 1.7);
        var fuji = p.Predict(
            track: "fuji", car: "KK-SII", condition: "dry",
            lapWithinStint: 5, ambientTempC: 18.0,
            trackTempC: null, cloudCoverPct: 50.0,
            corner: "fl", targetHotPressureBar: 1.7);
        Assert.Equal(tsk.KKelvinPerG2, fuji.KKelvinPerG2, precision: 9);
        Assert.Equal(tsk.TauSec, fuji.TauSec, precision: 9);
        // Schema v5 carries no track constant: the circuit enters through
        // its per-corner driving intensity instead.
        Assert.NotEqual(tsk.G2Typ, fuji.G2Typ);
        Assert.NotEqual(tsk.PredictedHotTempC, fuji.PredictedHotTempC);
    }

    // ---------- Schema v5: per-corner heat inputs ----------

    // Tiny synthetic v5 artifact: one car, one track, every corner sharing
    // the same K/tau so any hot-temp difference comes from q_typ_by_corner /
    // outlap_q_by_corner alone.
    private static TireModel SyntheticV5Model(bool withPerCorner)
    {
        var corners = new[] { "fl", "fr", "rl", "rr" };
        var dto = new TireModelDto(
            SchemaVersion: 5,
            FitAtUtc: "2026-10-08T00:00:00Z",
            ModelForm: "synthetic",
            GayLussac: new GayLussacConfigDto(1.0, 273.15, "T_air"),
            EnergyBalance: new EnergyBalanceConfigDto(0.2, false, new TRoadProxyConfigDto("x", 10.0, 1.0)),
            Conditions: new ConditionsConfigDto(new[] { "dry", "damp", "wet" }, "dry"),
            Corners: corners,
            MinSamplesPerBucket: 5,
            PriorsWhenNoFit: new PriorsDto(240.0, 60.0),
            TauSecByCarCornerCond: corners.Select(c =>
                new TauEntryDto("Car", c, "dry", 300.0, 0.0, 10, false)).ToList(),
            KBuckets: corners.Select(c =>
                new KBucketEntryDto(new KBucketKeyDto("Car", c, "dry"), 40.0, 0.0, 10, false, false)).ToList(),
            G2TypByTrackCarCond: new[]
            {
                new G2EntryDto("synth", "Car", "dry", 1.0, 10,
                    G2VsLapTime: new G2CurveDto(new[] { 50.0, 60.0, 70.0 }, new[] { 1.4, 1.0, 0.7 }, 10),
                    QTypByCorner: withPerCorner
                        ? new Dictionary<string, double> { ["fl"] = 1.3, ["fr"] = 0.9, ["rl"] = 1.1, ["rr"] = 0.7 }
                        : null),
            },
            LapTimeTypByTrackCarCond: new[] { new LapTimeEntryDto("synth", "Car", "dry", 60.0, 10) },
            OutlapTypByTrackCarCond: new[]
            {
                new OutlapEntryDto("synth", "Car", "dry", 90.0, 0.4, 10,
                    OutlapQByCorner: withPerCorner
                        ? new Dictionary<string, double> { ["fl"] = 0.6, ["fr"] = 0.3, ["rl"] = 0.5, ["rr"] = 0.2 }
                        : null),
            },
            G2LapTimeModel: new G2LapTimeModelDto("sector_curve", "x", 3.0, new MultiplierClampDto(0.4, 2.5)));
        return new TireModel(dto);
    }

    private static CornerPrediction PredictSynthetic(
        CircuitPredictor p, string corner, double? targetLapTimeS = null) => p.Predict(
            track: "synth", car: "Car", condition: "dry",
            lapWithinStint: 5, ambientTempC: 20.0,
            trackTempC: null, cloudCoverPct: 100.0,
            corner: corner, targetHotPressureBar: 1.8,
            targetLapTimeS: targetLapTimeS);

    [Fact]
    public void Predict_V5_PerCornerHeatInputsDriveTheCornerPrediction()
    {
        var p = new CircuitPredictor(SyntheticV5Model(withPerCorner: true));
        var fl = PredictSynthetic(p, "fl");
        var rr = PredictSynthetic(p, "rr");

        // Same K and tau: only the per-corner heat input differs.
        Assert.Equal(fl.KKelvinPerG2, rr.KKelvinPerG2);
        Assert.Equal(fl.TauSec, rr.TauSec);
        Assert.Equal(1.3, fl.G2Typ);
        Assert.Equal(0.7, rr.G2Typ);
        Assert.Equal(0.6, fl.OutlapG2);
        Assert.Equal(0.2, rr.OutlapG2);
        Assert.True(fl.PredictedHotTempC > rr.PredictedHotTempC, "FL (hotter corner) ends hotter");
        Assert.True(fl.ColdPressureBar < rr.ColdPressureBar);

        // The pace multiplier applies to the per-corner value exactly as to g2_typ.
        var flFast = PredictSynthetic(p, "fl", targetLapTimeS: 55.0);
        Assert.Equal("curve", flFast.G2PaceSource);
        Assert.True(flFast.G2Scale > 1.0);
        Assert.Equal(1.3 * flFast.G2Scale, flFast.G2Typ, precision: 12);
    }

    [Fact]
    public void Predict_V5_EntriesWithoutPerCornerFieldsFallBackToTheMean()
    {
        var p = new CircuitPredictor(SyntheticV5Model(withPerCorner: false));
        var fl = PredictSynthetic(p, "fl");
        var rr = PredictSynthetic(p, "rr");
        Assert.Equal(1.0, fl.G2Typ);
        Assert.Equal(1.0, rr.G2Typ);
        Assert.Equal(0.4, fl.OutlapG2);
        Assert.Equal(fl.PredictedHotTempC, rr.PredictedHotTempC);
        Assert.Equal(fl.ColdPressureBar, rr.ColdPressureBar);
    }

    // ---------- Fixture DTOs (private; only used by the parity test) ----------

    private sealed record FixtureCase(
        [property: System.Text.Json.Serialization.JsonPropertyName("label")] string Label,
        [property: System.Text.Json.Serialization.JsonPropertyName("inputs")] FixtureInputs Inputs,
        [property: System.Text.Json.Serialization.JsonPropertyName("corners")] Dictionary<string, FixtureCornerOutput> Corners,
        [property: System.Text.Json.Serialization.JsonPropertyName("g2_scale")] double? G2Scale = null,
        [property: System.Text.Json.Serialization.JsonPropertyName("g2_pace_source")] string? G2PaceSource = null);

    private sealed record FixtureInputs(
        [property: System.Text.Json.Serialization.JsonPropertyName("track")] string Track,
        [property: System.Text.Json.Serialization.JsonPropertyName("car")] string Car,
        [property: System.Text.Json.Serialization.JsonPropertyName("track_condition")] string TrackCondition,
        [property: System.Text.Json.Serialization.JsonPropertyName("lap_within_stint")] int LapWithinStint,
        [property: System.Text.Json.Serialization.JsonPropertyName("ambient_temp_c")] double AmbientTempC,
        [property: System.Text.Json.Serialization.JsonPropertyName("track_temp_c")] double? TrackTempC,
        [property: System.Text.Json.Serialization.JsonPropertyName("cloud_cover_pct")] double? CloudCoverPct,
        [property: System.Text.Json.Serialization.JsonPropertyName("cold_tire_temp_c")] double? ColdTireTempC,
        [property: System.Text.Json.Serialization.JsonPropertyName("target_hot_pressure_bar")] double TargetHotPressureBar,
        [property: System.Text.Json.Serialization.JsonPropertyName("target_lap_time_s")] double? TargetLapTimeS = null,
        [property: System.Text.Json.Serialization.JsonPropertyName("compound")] string? Compound = null);

    private sealed record FixtureCornerOutput(
        [property: System.Text.Json.Serialization.JsonPropertyName("cold_pressure_bar")] double ColdPressureBar,
        [property: System.Text.Json.Serialization.JsonPropertyName("predicted_hot_temp_c")] double PredictedHotTempC,
        [property: System.Text.Json.Serialization.JsonPropertyName("K_source_bucket")] List<string> KSourceBucket);
}
