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
                if (py.G2Scale is double expectedScale)
                {
                    Assert.True(Math.Abs(prediction.G2Scale - expectedScale) < 1e-9,
                        $"[{c.Label}/{corner}] g2 scale C#={prediction.G2Scale:F6} " +
                        $"vs Python={expectedScale:F6}");
                    Assert.Equal(c.G2PaceSource, prediction.G2PaceSource);
                }
                if (c.UsesSimulation is bool usesSimulation)
                {
                    Assert.Equal(c.Provenance?.Heat, prediction.HeatProvenance);
                    Assert.Equal(c.Provenance?.LapTime, prediction.LapTimeProvenance);
                    Assert.Equal(c.Provenance?.Outlap, prediction.OutlapProvenance);
                    Assert.Equal(usesSimulation, prediction.UsesSimulation);
                    Assert.Equal(c.PaceStatus, prediction.PaceStatus);
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
    private static TireModel SyntheticV5Model(bool withPerCorner, bool withCornerPace = false)
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
                    G2VsLapTime: new G2CurveDto(new[] { 50.0, 60.0, 70.0 }, new[] { 1.4, 1.0, 0.7 }, 10,
                        G2ByCorner: withCornerPace
                            ? new Dictionary<string, double[]> { ["fl"] = new[] { 2.0, 1.0, 0.5 } }
                            : null),
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
    public void Predict_PerCornerPaceCurveScalesEachCornerByItsOwnSeries()
    {
        // The simulated-lap curve carries one series per corner: a corner
        // with its own series uses it, a corner without falls back to the
        // corner-mean g2. Both anchor at 1 at the typical lap time.
        var p = new CircuitPredictor(SyntheticV5Model(withPerCorner: true, withCornerPace: true));
        var flFast = PredictSynthetic(p, "fl", targetLapTimeS: 50.0);
        var frFast = PredictSynthetic(p, "fr", targetLapTimeS: 50.0);
        Assert.Equal("curve", flFast.G2PaceSource);
        Assert.Equal(2.0, flFast.G2Scale, precision: 12);
        Assert.Equal(1.4, frFast.G2Scale, precision: 12);
        Assert.Equal(1.0, PredictSynthetic(p, "fl", targetLapTimeS: 60.0).G2Scale, precision: 12);
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

    // ---------- Logged entries per raw car (#383) ----------

    // SyntheticV5Model's "Car" pooling two raw cars, RawA and RawB: RawA has
    // its own dry entry at "synth" (heat, lap time, pace curve, out-lap),
    // RawB has none; the pooled car also has a damp entry. Same numbers as
    // the Python and web tests.
    private static TireModel RawCarModel(bool withRawTables = true)
    {
        var dto = SyntheticV5Model(withPerCorner: false).Dto;
        dto = dto with
        {
            CarAliases = new Dictionary<string, string> { ["RawA"] = "Car", ["RawB"] = "Car" },
            G2TypByTrackCarCond = dto.G2TypByTrackCarCond
                .Append(new G2EntryDto("synth", "Car", "damp", 0.6, 10)).ToList(),
            LapTimeTypByTrackCarCond = dto.LapTimeTypByTrackCarCond
                .Append(new LapTimeEntryDto("synth", "Car", "damp", 65.0, 10)).ToList(),
            G2TypByTrackRawCarCond = withRawTables
                ? new[]
                {
                    new G2EntryDto("synth", "RawA", "dry", 1.2, 4,
                        G2VsLapTime: new G2CurveDto(new[] { 50.0, 70.0 }, new[] { 2.0, 1.0 }, 4)),
                }
                : null,
            LapTimeTypByTrackRawCarCond = withRawTables
                ? new[] { new LapTimeEntryDto("synth", "RawA", "dry", 58.0, 4) }
                : null,
            OutlapTypByTrackRawCarCond = withRawTables
                ? new[] { new OutlapEntryDto("synth", "RawA", "dry", 80.0, 0.5, 2) }
                : null,
        };
        return new TireModel(dto);
    }

    private static CornerPrediction PredictRaw(
        CircuitPredictor p, string car, string condition = "dry", string track = "synth") => p.Predict(
            track: track, car: car, condition: condition,
            lapWithinStint: 5, ambientTempC: 20.0,
            trackTempC: null, cloudCoverPct: 100.0,
            corner: "fl", targetHotPressureBar: 1.8,
            targetLapTimeS: 60.0);

    [Fact]
    public void Predict_RawCar_ReadsItsOwnEntryAndThePooledConstants()
    {
        var p = new CircuitPredictor(RawCarModel());
        var own = PredictRaw(p, "RawA");
        var pooled = PredictRaw(p, "Car");
        // its own heat along its own curve, anchored at its own lap time (1.6 at 58 s)
        Assert.Equal(1.2 * 1.5 / 1.6, own.G2Typ, precision: 12);
        Assert.Equal(58.0, own.LapTimeTypS);
        Assert.Equal("curve", own.G2PaceSource);
        Assert.Equal(80.0, own.OutlapTimeS);
        Assert.Equal(0.5, own.OutlapG2);
        Assert.Equal(60.0, pooled.LapTimeTypS);
        Assert.Equal(90.0, pooled.OutlapTimeS);
        // the constants are the pooled car's whichever name is given
        Assert.Equal(pooled.KSourceBucket, own.KSourceBucket);
        Assert.Equal(pooled.KKelvinPerG2, own.KKelvinPerG2);
        Assert.Equal(pooled.TauSec, own.TauSec);
        // a raw car without its own entry reads the pooled one
        Assert.Equal(pooled, PredictRaw(p, "RawB"));
        // the condition chain comes first: RawA's damp is the pooled damp entry ...
        var damp = PredictRaw(p, "RawA", "damp");
        var pooledDamp = PredictRaw(p, "Car", "damp");
        Assert.Equal(65.0, damp.LapTimeTypS);
        Assert.Equal(pooledDamp.LapTimeTypS, damp.LapTimeTypS);
        // ... and each table walks the chain alone: no damp out-lap anywhere,
        // so the dry step reads RawA's own out-lap before the pooled one
        Assert.Equal("fallback(dry)", damp.OutlapSource);
        Assert.Equal(80.0, damp.OutlapTimeS);
        Assert.Equal(90.0, pooledDamp.OutlapTimeS);
    }

    [Fact]
    public void Predict_RawCar_WithoutTheTablesIsItsPooledCar()
    {
        var without = new CircuitPredictor(RawCarModel(withRawTables: false));
        foreach (var cond in new[] { "dry", "damp", "wet" })
            Assert.Equal(PredictRaw(without, "Car", cond), PredictRaw(without, "RawA", cond));
        // a car with no entry at the track averages the pooled rows only
        var withRaw = RawCarModel();
        var bare = RawCarModel(withRawTables: false);
        Assert.Equal(bare.LookupG2("synth", "Nobody", "dry"), withRaw.LookupG2("synth", "Nobody", "dry"));
        Assert.Equal("track_pooled", withRaw.LookupLapTime("synth", "Nobody", "dry").Source);
    }

    // ---------- Simulated entries (the cutover) ----------

    // RawCarModel with RawA's dry entry simulated (schema §9: beside the
    // logged rows, which stay): 14 knots from 50 s to 76 s, supported from
    // 54 s to 70 s, at a fed-use lap time of 57 s; its out-lap stays logged.
    // Same numbers as the web and Python tests.
    private static readonly double[] SimKnots = Enumerable.Range(0, 14).Select(i => 50.0 + 2 * i).ToArray();
    private const double SimTyp = 57.0;
    private static double[] SimSeries(double k) => SimKnots.Select((_, i) => k / (1 + 0.07 * i)).ToArray();

    private static TireModelDto SimModelDto(bool withSupport = true, bool simFirst = false)
    {
        var dto = RawCarModel().Dto;
        var byCorner = new Dictionary<string, double[]>
        {
            ["fl"] = SimSeries(2.6), ["fr"] = SimSeries(1.9), ["rl"] = SimSeries(2.3), ["rr"] = SimSeries(1.7),
        };
        var mean = SimKnots.Select((_, i) => byCorner.Values.Average(v => v[i])).ToArray();
        var curve = new G2CurveDto(SimKnots, mean, 0, byCorner,
            withSupport ? new PaceSupportDto(54.0, 70.0) : null);
        var heat = new G2EntryDto("synth", "RawA", "dry", 2.0, 0, curve,
            new Dictionary<string, double> { ["fl"] = 2.3, ["fr"] = 1.7, ["rl"] = 2.0, ["rr"] = 1.5 },
            Source: "sim");
        var lap = new LapTimeEntryDto("synth", "RawA", "dry", SimTyp, 0, Source: "sim");
        IReadOnlyList<T> Beside<T>(IReadOnlyList<T>? rows, T row) =>
            simFirst ? new[] { row }.Concat(rows ?? Array.Empty<T>()).ToList()
                     : (rows ?? Array.Empty<T>()).Append(row).ToList();
        return dto with
        {
            G2TypByTrackRawCarCond = Beside(dto.G2TypByTrackRawCarCond, heat),
            LapTimeTypByTrackRawCarCond = Beside(dto.LapTimeTypByTrackRawCarCond, lap),
        };
    }

    private static double Linear(double x, IReadOnlyList<double> xs, IReadOnlyList<double> ys)
    {
        int i = 1;
        while (xs[i] < x) i++;
        return ys[i - 1] + (x - xs[i - 1]) / (xs[i] - xs[i - 1]) * (ys[i] - ys[i - 1]);
    }

    private static CornerPrediction PredictSim(
        CircuitPredictor p, string car, double? targetLapTimeS, string condition = "dry",
        string corner = "fl", bool includeOutlap = true) => p.Predict(
            track: "synth", car: car, condition: condition,
            lapWithinStint: 5, ambientTempC: 20.0,
            trackTempC: null, cloudCoverPct: 100.0,
            corner: corner, targetHotPressureBar: 1.8,
            targetLapTimeS: targetLapTimeS, includeOutlap: includeOutlap);

    [Fact]
    public void Predict_SimulatedEntry_ProvenanceKnotsAndPaceFlag()
    {
        var model = new TireModel(SimModelDto());
        var p = new CircuitPredictor(model);
        var hit = PredictSim(p, "RawA", 61.0);
        Assert.Equal("sim", hit.HeatProvenance);
        Assert.Equal("sim", hit.LapTimeProvenance);
        Assert.Equal("logged", hit.OutlapProvenance);
        Assert.True(hit.UsesSimulation);
        Assert.Equal("ok", hit.PaceStatus);
        // the fed-use lap time is the prefill, and the 14 knots are read linearly
        Assert.Equal(SimTyp, model.LookupLapTime("synth", "RawA", "dry").ValueSeconds);
        var fl = SimSeries(2.6);
        Assert.Equal(2.3 * Linear(61.0, SimKnots, fl) / Linear(SimTyp, SimKnots, fl), hit.G2Typ, precision: 12);
        // no target: the typical lap time's status
        Assert.Equal("ok", PredictSim(p, "RawA", null).PaceStatus);
        // outside the logged pace, either side, or past the slowest knot: answered, flagged
        foreach (var t in new[] { 52.0, 71.0, 80.0 })
        {
            var q = PredictSim(p, "RawA", t);
            Assert.Equal("outside_support", q.PaceStatus);
            Assert.True(double.IsFinite(q.ColdPressureBar));
        }
        // faster than the fastest knot: answered at it (the clamp), flagged
        var fast = PredictSim(p, "RawA", 47.0);
        Assert.Equal("no_solution", fast.PaceStatus);
        Assert.Equal(fl[0] / Linear(SimTyp, SimKnots, fl), fast.G2Scale, precision: 12);
        Assert.Equal(5 * 47.0, fast.TAtLapNs);
        // a sim curve without its support is outside it
        var bare = new CircuitPredictor(new TireModel(SimModelDto(withSupport: false)));
        Assert.Equal("outside_support", PredictSim(bare, "RawA", 61.0).PaceStatus);
        // no out-lap asked for: none read
        Assert.Null(PredictSim(p, "RawA", null, includeOutlap: false).OutlapProvenance);
    }

    [Fact]
    public void Predict_ASimRowAnswersItsExactConditionOnly_WhateverItsPlaceInTheTable()
    {
        foreach (var simFirst in new[] { false, true })
        {
            var model = new TireModel(SimModelDto(simFirst: simFirst));
            Assert.Equal("sim", PredictSim(new CircuitPredictor(model), "RawA", 61.0).HeatProvenance);
            Assert.Equal(SimTyp, model.LookupLapTime("synth", "RawA", "dry").ValueSeconds);
        }
        // with the pooled damp rows gone, RawA's wet and damp fall back to its
        // own dry rows: the logged ones, as today, not the simulated one beside
        TireModelDto Strip(TireModelDto d) => d with
        {
            G2TypByTrackCarCond = d.G2TypByTrackCarCond.Where(r => r.Condition != "damp").ToList(),
            LapTimeTypByTrackCarCond = d.LapTimeTypByTrackCarCond.Where(r => r.Condition != "damp").ToList(),
        };
        var todayDry = new CircuitPredictor(new TireModel(Strip(RawCarModel().Dto)));
        var simDry = new TireModel(Strip(SimModelDto()));
        foreach (var cond in new[] { "wet", "damp" })
        {
            var q = PredictSim(new CircuitPredictor(simDry), "RawA", 61.0, condition: cond);
            Assert.Equal(PredictSim(todayDry, "RawA", 61.0, condition: cond), q);
            Assert.Equal("curve", q.G2PaceSource);
            Assert.False(q.UsesSimulation);
            Assert.Equal("fallback(dry)", simDry.LookupG2("synth", "RawA", cond).Source);
        }
        // with them, the pooled damp entry, as today; and a saved pooled car
        // reads the pooled logged rows
        var today = new CircuitPredictor(RawCarModel());
        var sim = new CircuitPredictor(new TireModel(SimModelDto()));
        foreach (var cond in new[] { "wet", "damp" })
            Assert.Equal(PredictSim(today, "RawA", 61.0, condition: cond), PredictSim(sim, "RawA", 61.0, condition: cond));
        foreach (var cond in new[] { "dry", "damp", "wet" })
            Assert.Equal(PredictSim(today, "Car", 61.0, condition: cond), PredictSim(sim, "Car", 61.0, condition: cond));
        // the exact step's order is raw sim, raw logged, pooled sim, pooled
        // logged: a pooled sim row does not beat the raw car's own logged
        // row, and a raw car with no row of its own reads the pooled sim row first
        var simDto = SimModelDto();
        var simRow = simDto.G2TypByTrackRawCarCond!.Single(r => r.Source == "sim");
        var ordered = new TireModel(simDto with
        {
            G2TypByTrackRawCarCond = simDto.G2TypByTrackRawCarCond!.Where(r => r.Source != "sim").ToList(),
            G2TypByTrackCarCond = simDto.G2TypByTrackCarCond.Prepend(simRow with { Car = "Car" }).ToList(),
        });
        Assert.Equal(new G2Lookup(1.2, 4, "exact"), ordered.LookupG2("synth", "RawA", "dry"));
        Assert.Equal("sim", TireModel.RowProvenance(ordered.HeatRows("synth", "RawB", "dry").Rows));
        Assert.Equal("logged", TireModel.RowProvenance(ordered.HeatRows("synth", "RawB", "wet").Rows));
    }

    [Fact]
    public void Predict_LoggedRowsCarryNoAttribution_AndAveragesReadLoggedRowsOnly()
    {
        var dto = SimModelDto();
        var p = new CircuitPredictor(new TireModel(dto));
        foreach (var car in new[] { "Car", "RawB" })
        {
            var q = PredictSim(p, car, 61.0, corner: "rr");
            Assert.Equal(("logged", "logged", "logged", false, (string?)null),
                (q.HeatProvenance, q.LapTimeProvenance, q.OutlapProvenance!, q.UsesSimulation, q.PaceStatus));
        }
        // a simulated out-lap alone carries the attribution
        var raw = RawCarModel().Dto;
        var outSim = raw with
        {
            OutlapTypByTrackRawCarCond = raw.OutlapTypByTrackRawCarCond!
                .Append(raw.OutlapTypByTrackRawCarCond!.Single() with { Source = "sim", OutlapMovingS = 70.0 }).ToList(),
        };
        var o = PredictSim(new CircuitPredictor(new TireModel(outSim)), "RawA", 61.0);
        Assert.Equal(("logged", "sim", 70.0, true, (string?)null),
            (o.HeatProvenance, o.OutlapProvenance!, o.OutlapTimeS, o.UsesSimulation, o.PaceStatus));
        // an averaged fallback reads logged rows only (the lead's ruling of
        // 2026-11-27): the simulated damp row is not counted ...
        Assert.Equal((1.0 + 0.6) / 2, new TireModel(dto).LookupG2("synth", "Nobody", "dry").Value, precision: 12);
        var pooledSim = dto with
        {
            G2TypByTrackCarCond = dto.G2TypByTrackCarCond
                .Select(r => r.Condition == "damp" ? r with { Source = "sim" } : r).ToList(),
        };
        var pooled = new TireModel(pooledSim);
        var nobody = PredictSim(new CircuitPredictor(pooled), "Nobody", 61.0);
        Assert.Equal(new G2Lookup(1.0, 10, "track_pooled"), pooled.LookupG2("synth", "Nobody", "dry"));
        Assert.Equal("logged", nobody.HeatProvenance);
        Assert.Null(nobody.PaceStatus);
        // ... a car whose rows at the track are all simulated skips to the
        // track's logged rows, and a track with none to the global logged mean
        var onlySim = pooledSim with
        {
            G2TypByTrackCarCond = pooledSim.G2TypByTrackCarCond.Where(r => r.Condition != "dry")
                .Append(new G2EntryDto("other", "Car", "dry", 0.5, 3)).ToList(),
        };
        var skip = new TireModel(onlySim);
        Assert.Equal(new G2Lookup(0.5, 0, "global"), skip.LookupG2("synth", "Car", "dry"));
        Assert.Equal(new G2Lookup(0.5, 0, "global"), skip.LookupG2("synth", "Nobody", "dry"));
        var lapsSim = onlySim with
        {
            LapTimeTypByTrackCarCond = onlySim.LapTimeTypByTrackCarCond.Select(r => r with { Source = "sim" }).ToList(),
        };
        Assert.Equal(new LapTimeLookup(90.0, 0, "global"), new TireModel(lapsSim).LookupLapTime("synth", "Nobody", "dry"));
    }

    [Fact]
    public void AvailableTracks_IncludesATrackOnlyARawCarsRowsCover()
    {
        var dto = SimModelDto();
        Assert.Equal(new[] { "synth" }, new TireModel(dto).AvailableTracks);
        var own = dto.G2TypByTrackRawCarCond!.Single(r => r.Source == "sim");
        var model = new TireModel(dto with
        {
            G2TypByTrackRawCarCond = dto.G2TypByTrackRawCarCond!.Append(own with { TrackCanonical = "simonly" }).ToList(),
        });
        Assert.Equal(new[] { "simonly", "synth" }, model.AvailableTracks);
        var q = new CircuitPredictor(model).Predict("simonly", "RawA", "dry", 5, 20.0, null, null, "fl", 1.8);
        Assert.Equal("sim", q.HeatProvenance);
        Assert.Equal(2.3, q.G2Typ);
    }

    // The bundled artifact as it was before the cutover: its simulated rows
    // taken out and no row carrying a source.
    private static TireModel WithoutSimulation(TireModel model)
    {
        static List<T>? Logged<T>(IReadOnlyList<T>? rows, Func<T, T> unflag) where T : class, IEntryRow =>
            rows?.Where(r => r.Source != TireModel.Sim).Select(unflag).ToList();
        var d = model.Dto;
        return new TireModel(d with
        {
            G2TypByTrackCarCond = Logged(d.G2TypByTrackCarCond, r => r with { Source = null })!,
            G2TypByTrackRawCarCond = Logged(d.G2TypByTrackRawCarCond, r => r with { Source = null }),
            LapTimeTypByTrackCarCond = Logged(d.LapTimeTypByTrackCarCond, r => r with { Source = null })!,
            LapTimeTypByTrackRawCarCond = Logged(d.LapTimeTypByTrackRawCarCond, r => r with { Source = null }),
            OutlapTypByTrackCarCond = Logged(d.OutlapTypByTrackCarCond, r => r with { Source = null }),
            OutlapTypByTrackRawCarCond = Logged(d.OutlapTypByTrackRawCarCond, r => r with { Source = null }),
        });
    }

    [Fact]
    public void Predict_AnArtifactWithoutSourceFlags_ReadsAsLoggedEverywhere()
    {
        var p = new CircuitPredictor(WithoutSimulation(LoadPredictor().Model));
        foreach (var track in p.Model.AvailableTracks)
        foreach (var car in p.Model.AvailableCars)
        foreach (var cond in p.Model.AvailableConditions)
        {
            var q = p.Predict(track, car, cond, 5, 18.0, null, null, "fl", 1.7);
            Assert.False(q.UsesSimulation, $"{track}/{car}/{cond}");
            Assert.Null(q.PaceStatus);
        }
    }

    [Fact]
    public void Predict_TheBundledArtifactsLoggedCells_ReadAsTheyWouldWithoutTheSimulatedRows()
    {
        // the simulated rows sit beside the logged ones and answer only their
        // own cell: wherever the bundled artifact answers from logged rows, it
        // is the artifact without them, to the last field (schema §9)
        var bundled = LoadPredictor();
        var legacy = new CircuitPredictor(WithoutSimulation(bundled.Model));
        foreach (var track in bundled.Model.AvailableTracks.Intersect(legacy.Model.AvailableTracks))
        foreach (var car in bundled.Model.AvailableCars)
        foreach (var cond in bundled.Model.AvailableConditions)
        {
            var q = bundled.Predict(track, car, cond, 5, 18.0, null, null, "rl", 1.7);
            if (q.UsesSimulation) continue;
            Assert.Equal(legacy.Predict(track, car, cond, 5, 18.0, null, null, "rl", 1.7), q);
        }
    }

    // The kind of row a lookup answers a cell with, as the artifact's
    // entry_coverage states it (sim_entries.answer): the first row the
    // condition chain finds ("sim" / "logged"; a curve-less one is "none"
    // for the curve), else "averaged" (a track or global mean), else "none".
    // Mirrors the web test.
    private static string AnswerKind<T>((List<T> Rows, string Step) found, bool curve = false)
        where T : class, IEntryRow
    {
        if (found.Rows.Count == 0) return "none";
        if (found.Step is "track_car_pooled" or "track_pooled" or "global") return "averaged";
        if (curve && found.Rows[0] is G2EntryDto { G2VsLapTime: null }) return "none";
        return found.Rows[0].Source ?? TireModel.Logged;
    }

    private static JsonDocument EmbeddedModelJson()
    {
        var stream = typeof(TireModel).Assembly.GetManifestResourceStream("tire_model.json")
            ?? throw new InvalidDataException("no embedded tire_model.json");
        using (stream) return JsonDocument.Parse(stream);
    }

    [Fact]
    public void EntryCoverage_IsWhatTheLookupsAnswer()
    {
        using var doc = EmbeddedModelJson();
        // An artifact from before the cutover has no coverage to check.
        if (!doc.RootElement.TryGetProperty("entry_coverage", out var cells)) return;
        var p = LoadPredictor();
        var model = p.Model;
        var offered = (from t in model.AvailableTracks
                       from c in model.AvailableCars
                       from k in model.AvailableConditions
                       select $"{t}|{c}|{k}").ToHashSet();
        var listed = new HashSet<string>();
        foreach (var cell in cells.EnumerateArray())
        {
            string Get(string key) => cell.GetProperty(key).GetString()!;
            var (track, car, cond) = (Get("track_canonical"), Get("car"), Get("condition"));
            var label = $"{track}/{car}/{cond}";
            listed.Add($"{track}|{car}|{cond}");
            var heat = model.HeatRows(track, car, cond);
            Assert.True(Get("heat") == AnswerKind(heat), $"{label}: heat");
            Assert.True(Get("curve") == AnswerKind(heat, curve: true), $"{label}: curve");
            Assert.True(Get("lap_time") == AnswerKind(model.LapTimeRows(track, car, cond)), $"{label}: lap time");
            Assert.True(Get("outlap") == AnswerKind(model.OutlapRows(track, car, cond)), $"{label}: out-lap");
            var q = p.Predict(track, car, cond, 5, 18.0, null, null, "fl", 1.7);
            Assert.True(q.UsesSimulation == new[] { Get("heat"), Get("lap_time"), Get("outlap") }.Contains("sim"),
                $"{label}: attribution");
        }
        Assert.True(offered.SetEquals(listed), "entry_coverage lists every offered cell, and only those");
    }

    [Fact]
    public void ArtifactAttribution_IsTheUiString()
    {
        using var doc = EmbeddedModelJson();
        if (!doc.RootElement.TryGetProperty("simulation", out var sim)
            || !sim.TryGetProperty("attribution", out var attribution)) return;
        var localizer = TirePressureCalculator.Localization.Localizer.Instance;
        foreach (var lang in new[] { "en", "ja" })
        {
            localizer.SetPreference(lang);
            Assert.Equal(attribution.GetProperty(lang).GetString(), localizer["GsiAttribution"]);
        }
        localizer.SetPreference("en");
    }

    // ---------- Fixture DTOs (private; only used by the parity test) ----------

    private sealed record FixtureCase(
        [property: System.Text.Json.Serialization.JsonPropertyName("label")] string Label,
        [property: System.Text.Json.Serialization.JsonPropertyName("inputs")] FixtureInputs Inputs,
        [property: System.Text.Json.Serialization.JsonPropertyName("corners")] Dictionary<string, FixtureCornerOutput> Corners,
        [property: System.Text.Json.Serialization.JsonPropertyName("g2_pace_source")] string? G2PaceSource = null,
        [property: System.Text.Json.Serialization.JsonPropertyName("provenance")] FixtureProvenance? Provenance = null,
        [property: System.Text.Json.Serialization.JsonPropertyName("uses_simulation")] bool? UsesSimulation = null,
        [property: System.Text.Json.Serialization.JsonPropertyName("pace_status")] string? PaceStatus = null);

    private sealed record FixtureProvenance(
        [property: System.Text.Json.Serialization.JsonPropertyName("heat")] string Heat,
        [property: System.Text.Json.Serialization.JsonPropertyName("lap_time")] string LapTime,
        [property: System.Text.Json.Serialization.JsonPropertyName("outlap")] string? Outlap);

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
        [property: System.Text.Json.Serialization.JsonPropertyName("K_source_bucket")] List<string> KSourceBucket,
        [property: System.Text.Json.Serialization.JsonPropertyName("g2_scale")] double? G2Scale = null);
}
