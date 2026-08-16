using System;
using System.Collections.Generic;
using System.Linq;

namespace TirePressureCalculator.Services.Modeling;

/// <summary>
/// In-memory wrapper around a parsed <see cref="TireModelDto"/>. Adds
/// lookup helpers + the condition fallback chain that mirrors the Python
/// predictor at <c>src/motorsports_data_notebook/tire_model/predict.py</c>.
/// </summary>
public sealed class TireModel
{
    // v3 adds the target-lap-time feature; v2 artifacts still load (the
    // pace scaling then always uses the exponent fallback defaults).
    public const int SupportedSchemaVersion = 3;
    public const int MinSupportedSchemaVersion = 2;

    public TireModelDto Dto { get; }

    public TireModel(TireModelDto dto)
    {
        if (dto.SchemaVersion < MinSupportedSchemaVersion || dto.SchemaVersion > SupportedSchemaVersion)
        {
            throw new InvalidOperationException(
                $"Unsupported tire_model.json schema_version {dto.SchemaVersion}; expected {MinSupportedSchemaVersion}–{SupportedSchemaVersion}. " +
                "Rebuild the artifact with `just tire-build-warmup-table`.");
        }
        Dto = dto;
    }

    public IReadOnlyList<string> AvailableCars => Dto.TauSecByCarCornerCond
        .Select(r => r.Car).Distinct().OrderBy(s => s).ToList();

    // Every track with observed data, not just those with a fitted c_track —
    // thin tracks predict via the c_track prior until enough laps accumulate.
    public IReadOnlyList<string> AvailableTracks => Dto.CTrackByTrack
        .Select(r => r.TrackCanonical)
        .Concat(Dto.G2TypByTrackCarCond.Select(r => r.TrackCanonical))
        .Distinct().OrderBy(s => s).ToList();

    public IReadOnlyList<string> AvailableConditions => Dto.Conditions.Values;

    public string DefaultCondition => Dto.Conditions.Default;

    public double WRoad => Dto.EnergyBalance.WRoad;

    public double SunFactorDefault => Dto.EnergyBalance.TRoadProxy.SunFactorDefault;

    public double DeltaSunMaxC => Dto.EnergyBalance.TRoadProxy.DeltaSunMaxC;

    public double PAtmBar => Dto.GayLussac.PAtmBar;

    public double TZeroCToK => Dto.GayLussac.TZeroCToK;

    // ---- Condition fallback chain (mirrors predict.py:_CONDITION_FALLBACK) ----

    private static readonly IReadOnlyDictionary<string, IReadOnlyList<string>> _conditionChain =
        new Dictionary<string, IReadOnlyList<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["dry"] = new[] { "dry" },
            ["damp"] = new[] { "damp", "dry" },
            ["wet"] = new[] { "wet", "damp", "dry" },
        };

    internal static IReadOnlyList<string> ConditionChain(string condition) =>
        _conditionChain.TryGetValue(condition, out var chain)
            ? chain
            : new[] { condition, "dry" };

    // ---- Lookups ----

    public TauLookup LookupTau(string car, string corner, string condition)
    {
        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.TauSecByCarCornerCond.FirstOrDefault(
                r => r.Car == car && r.Corner == corner && r.Condition == cond);
            if (hit is not null)
            {
                return new TauLookup(hit.ValueSeconds, hit.StderrSeconds,
                    SourceBucket: $"({car}, {corner}, {cond})", FromPrior: hit.FromPrior);
            }
        }
        // (car, corner) mean across conditions
        var sameCC = Dto.TauSecByCarCornerCond.Where(r => r.Car == car && r.Corner == corner).ToList();
        if (sameCC.Count > 0)
        {
            return new TauLookup(sameCC.Average(r => r.ValueSeconds), 0.0,
                SourceBucket: $"({car}, {corner})", FromPrior: false);
        }
        // (car) mean across all corners + conditions
        var sameCar = Dto.TauSecByCarCornerCond.Where(r => r.Car == car).ToList();
        if (sameCar.Count > 0)
        {
            return new TauLookup(sameCar.Average(r => r.ValueSeconds), 0.0,
                SourceBucket: $"({car})", FromPrior: false);
        }
        return new TauLookup(Dto.PriorsWhenNoFit.TauSecSeconds, 0.0,
            SourceBucket: "(prior)", FromPrior: true);
    }

    public KLookup LookupK(string car, string corner, string condition)
    {
        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.KBuckets.FirstOrDefault(
                r => r.Key.Car == car && r.Key.Corner == corner && r.Key.Condition == cond);
            if (hit is not null)
            {
                return new KLookup(hit.ValueKelvinPerG2, hit.StderrKelvinPerG2, hit.NSamples,
                    SourceBucket: $"({car}, {corner}, {cond})", FromPrior: hit.FromPrior);
            }
        }
        var sameCC = Dto.KBuckets.Where(r => r.Key.Car == car && r.Key.Corner == corner).ToList();
        if (sameCC.Count > 0)
        {
            return new KLookup(sameCC.Average(r => r.ValueKelvinPerG2), 0.0,
                sameCC.Sum(r => r.NSamples),
                SourceBucket: $"({car}, {corner})", FromPrior: false);
        }
        var sameCar = Dto.KBuckets.Where(r => r.Key.Car == car).ToList();
        if (sameCar.Count > 0)
        {
            return new KLookup(sameCar.Average(r => r.ValueKelvinPerG2), 0.0,
                sameCar.Sum(r => r.NSamples),
                SourceBucket: $"({car})", FromPrior: false);
        }
        return new KLookup(Dto.PriorsWhenNoFit.KKelvinPerG2, 0.0, 0,
            SourceBucket: "(prior)", FromPrior: true);
    }

    public CTrackLookup LookupCTrack(string track)
    {
        var hit = Dto.CTrackByTrack.FirstOrDefault(r => r.TrackCanonical == track);
        if (hit is not null)
            return new CTrackLookup(hit.Value, hit.Stderr, FromPrior: false);
        return new CTrackLookup(Dto.PriorsWhenNoFit.CTrack, 0.0, FromPrior: true);
    }

    public G2Lookup LookupG2(string track, string car, string condition)
    {
        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.G2TypByTrackCarCond.FirstOrDefault(
                r => r.TrackCanonical == track && r.Car == car && r.Condition == cond);
            if (hit is not null)
            {
                var tag = cond == condition ? "exact" : $"fallback({cond})";
                return new G2Lookup(hit.G2Typ, hit.NLapsUsed, tag);
            }
        }
        var sameTC = Dto.G2TypByTrackCarCond.Where(
            r => r.TrackCanonical == track && r.Car == car).ToList();
        if (sameTC.Count > 0)
        {
            return new G2Lookup(sameTC.Average(r => r.G2Typ),
                sameTC.Sum(r => r.NLapsUsed), "track_car_pooled");
        }
        var sameT = Dto.G2TypByTrackCarCond.Where(r => r.TrackCanonical == track).ToList();
        if (sameT.Count > 0)
        {
            return new G2Lookup(sameT.Average(r => r.G2Typ),
                sameT.Sum(r => r.NLapsUsed), "track_pooled");
        }
        if (Dto.G2TypByTrackCarCond.Count > 0)
        {
            return new G2Lookup(Dto.G2TypByTrackCarCond.Average(r => r.G2Typ), 0, "global");
        }
        return new G2Lookup(0.7, 0, "global");
    }

    // ---- Target-lap-time pace scaling (schema v3) ----

    /// <summary>Piecewise-linear interpolation clamped to the endpoints.
    /// Must stay in lockstep with the Python and web implementations
    /// (pinned by the parity fixture).</summary>
    internal static double InterpClamped(double x, IReadOnlyList<double> xs, IReadOnlyList<double> ys)
    {
        if (x <= xs[0]) return ys[0];
        if (x >= xs[xs.Count - 1]) return ys[ys.Count - 1];
        for (int i = 1; i < xs.Count; i++)
        {
            if (x <= xs[i])
            {
                double w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
                return ys[i - 1] + w * (ys[i] - ys[i - 1]);
            }
        }
        return ys[ys.Count - 1];
    }

    private G2CurveDto? LookupG2PaceCurve(string track, string car, string condition)
    {
        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.G2TypByTrackCarCond.FirstOrDefault(
                r => r.TrackCanonical == track && r.Car == car && r.Condition == cond);
            if (hit is not null) return hit.G2VsLapTime;
        }
        return null;
    }

    /// <summary>
    /// Multiplier on g2_typ for a target lap time; mirrors the Python
    /// predictor's <c>_g2_pace_scale</c>. Preferred: ratio along the
    /// bucket's sector-fit curve (anchored at lap_time_typ so
    /// target == typical scales by exactly 1). Fallback: the pooled
    /// sector exponent. Clamped either way.
    /// </summary>
    public G2PaceScale ComputeG2PaceScale(
        string track, string car, string condition,
        double lapTimeTypS, double targetLapTimeS)
    {
        double clampMin = Dto.G2LapTimeModel?.MultiplierClamp?.Min ?? 0.4;
        double clampMax = Dto.G2LapTimeModel?.MultiplierClamp?.Max ?? 2.5;

        var curve = LookupG2PaceCurve(track, car, condition);
        if (curve is not null)
        {
            double reference = InterpClamped(lapTimeTypS, curve.LapTimeS, curve.G2);
            if (reference > 0)
            {
                double curveScale = InterpClamped(targetLapTimeS, curve.LapTimeS, curve.G2) / reference;
                return new G2PaceScale(
                    Math.Min(clampMax, Math.Max(clampMin, curveScale)), "curve");
            }
        }

        double exponent = Dto.G2LapTimeModel?.DefaultExponent ?? 3.0;
        double scale = Math.Pow(lapTimeTypS / targetLapTimeS, exponent);
        return new G2PaceScale(Math.Min(clampMax, Math.Max(clampMin, scale)), "exponent");
    }

    public LapTimeLookup LookupLapTime(string track, string car, string condition)
    {
        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.LapTimeTypByTrackCarCond.FirstOrDefault(
                r => r.TrackCanonical == track && r.Car == car && r.Condition == cond);
            if (hit is not null)
            {
                var tag = cond == condition ? "exact" : $"fallback({cond})";
                return new LapTimeLookup(hit.LapTimeTypS, hit.NLapsUsed, tag);
            }
        }
        var sameTC = Dto.LapTimeTypByTrackCarCond.Where(
            r => r.TrackCanonical == track && r.Car == car).ToList();
        if (sameTC.Count > 0)
        {
            return new LapTimeLookup(sameTC.Average(r => r.LapTimeTypS),
                sameTC.Sum(r => r.NLapsUsed), "track_car_pooled");
        }
        var sameT = Dto.LapTimeTypByTrackCarCond.Where(r => r.TrackCanonical == track).ToList();
        if (sameT.Count > 0)
        {
            return new LapTimeLookup(sameT.Average(r => r.LapTimeTypS),
                sameT.Sum(r => r.NLapsUsed), "track_pooled");
        }
        return new LapTimeLookup(90.0, 0, "global");
    }
}

public readonly record struct TauLookup(
    double ValueSeconds, double StderrSeconds, string SourceBucket, bool FromPrior);

public readonly record struct KLookup(
    double ValueKelvinPerG2, double StderrKelvinPerG2, int NSamples, string SourceBucket, bool FromPrior);

public readonly record struct CTrackLookup(
    double Value, double Stderr, bool FromPrior);

public readonly record struct G2Lookup(
    double Value, int NLapsUsed, string Source);

public readonly record struct LapTimeLookup(
    double ValueSeconds, int NLapsUsed, string Source);

public readonly record struct G2PaceScale(
    double Scale, string Source);
