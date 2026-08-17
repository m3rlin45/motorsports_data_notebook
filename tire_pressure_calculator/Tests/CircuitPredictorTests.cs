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
    public void Predict_WetWithNoWetData_FallsBackToDamp()
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
        // No (KK-SII, fl, wet) entry exists → chain falls back to damp.
        Assert.Contains("damp", result.KSourceBucket);
    }

    [Fact]
    public void Predict_ColdTireTempOverride_OnlyMovesGayLussacColdSide()
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

        // T_eff and predicted hot temp must not change.
        Assert.Equal(noOverride.TEffC, warmTire.TEffC, precision: 9);
        Assert.Equal(noOverride.PredictedHotTempC, warmTire.PredictedHotTempC, precision: 9);
        // T_cold differs.
        Assert.Equal(15.0, noOverride.TColdC, precision: 9);
        Assert.Equal(25.0, warmTire.TColdC, precision: 9);
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
    public void Predict_CrossTrack_UsesSameKDifferentCTrack()
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
        Assert.NotEqual(tsk.CTrack, fuji.CTrack);
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
