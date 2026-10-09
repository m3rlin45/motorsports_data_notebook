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
    // v5 drops the per-track constant c_track (the circuit enters only
    // through its per-corner heat input q_typ_by_corner / outlap_q_by_corner).
    // Older artifacts carried K values fitted against c_track and are no
    // longer loadable.
    public const int SupportedSchemaVersion = 5;
    public const int MinSupportedSchemaVersion = 5;

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

    /// <summary>
    /// Newest session that fed the fit, for the footer: the display-ready
    /// track-local start time ("2026-10-02 14:23 JST") when the artifact has
    /// it, else its date, else (older artifacts) the fit timestamp's date.
    /// </summary>
    public string? DataThrough =>
        !string.IsNullOrEmpty(Dto.DataThroughLocal) ? Dto.DataThroughLocal
        : Dto.DataThroughDate is { Length: >= 10 } d ? d[..10]
        : Dto.FitAtUtc is { Length: >= 10 } iso ? iso[..10]
        : null;

    public IReadOnlyList<string> AvailableCars => Dto.TauSecByCarCornerCond
        .Select(r => r.Car).Distinct().OrderBy(s => s).ToList();

    /// <summary>Resolve an alias-pooled car name (e.g. KK-F / KK-SII → FJ)
    /// onto the label the model was fitted with. Unknown cars pass through.</summary>
    public string ResolveCar(string car) =>
        Dto.CarAliases is { } aliases && aliases.TryGetValue(car, out var pooled) ? pooled : car;

    // Every track with observed data (a typical heat input).
    public IReadOnlyList<string> AvailableTracks => Dto.G2TypByTrackCarCond
        .Select(r => r.TrackCanonical)
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

    // ---- Compound-aware K (base × compound multiplier) ----

    /// <summary>Distinct compounds fitted for a car, for UI enumeration.
    /// One tire runs on all four corners — the choice is forced, so an
    /// empty list means the car predates compound labeling.</summary>
    public IReadOnlyList<string> AvailableCompounds(string car) =>
        (Dto.KByCarCompoundCornerCond ?? Array.Empty<KCompoundEntryDto>())
            .Where(r => r.Car == car).Select(r => r.Compound)
            .Distinct().OrderBy(s => s).ToList();

    /// <summary>Compound-specific K via the condition chain, or null when
    /// the artifact has no fitted bucket (caller falls back to pooled K).</summary>
    public KLookup? LookupCompoundK(string car, string compound, string corner, string condition)
    {
        var rows = Dto.KByCarCompoundCornerCond ?? Array.Empty<KCompoundEntryDto>();
        foreach (var cond in ConditionChain(condition))
        {
            var hit = rows.FirstOrDefault(
                r => r.Car == car && r.Compound == compound
                     && r.Corner == corner && r.Condition == cond);
            if (hit is not null)
            {
                return new KLookup(hit.ValueKelvinPerG2, hit.StderrKelvinPerG2, hit.NLaps,
                    SourceBucket: $"({car}, {compound}, {corner}, {cond})", FromPrior: false);
            }
        }
        return null;
    }

    /// <summary>Steady-state hot temp / hot pressure medians for UI
    /// prefills, via the condition chain; null when the artifact has no
    /// entry (caller keeps its static defaults).</summary>
    public CornerDefaultsLookup? LookupCornerDefaults(string car, string corner, string condition)
    {
        var rows = Dto.CornerDefaultsByCarCornerCond ?? Array.Empty<CornerDefaultsEntryDto>();
        foreach (var cond in ConditionChain(condition))
        {
            var hit = rows.FirstOrDefault(
                r => r.Car == car && r.Corner == corner && r.Condition == cond);
            if (hit is not null)
            {
                return new CornerDefaultsLookup(hit.HotTempC, hit.HotPressureBar, hit.NLapsUsed,
                    Source: cond == condition ? "exact" : $"fallback({cond})");
            }
        }
        return null;
    }

    /// <summary>Per-corner value from a v5 map, else the entry's corner
    /// mean. A null corner (or a map without that corner) keeps the mean,
    /// so pre-v5 artifacts produce identical numbers.</summary>
    private static double PerCornerOr(
        IReadOnlyDictionary<string, double>? byCorner, string? corner, double mean) =>
        corner is not null && byCorner is not null && byCorner.TryGetValue(corner, out var v)
            ? v
            : mean;

    /// <summary>
    /// Typical flying-lap heat input (g² in v2–v4; the per-corner q_typ in
    /// v5 when <paramref name="corner"/> is given and the entry carries
    /// <c>q_typ_by_corner</c>). Pooled fallbacks average the same per-corner
    /// value over the pooled rows.
    /// </summary>
    public G2Lookup LookupG2(string track, string car, string condition, string? corner = null)
    {
        double Value(G2EntryDto r) => PerCornerOr(r.QTypByCorner, corner, r.G2Typ);

        foreach (var cond in ConditionChain(condition))
        {
            var hit = Dto.G2TypByTrackCarCond.FirstOrDefault(
                r => r.TrackCanonical == track && r.Car == car && r.Condition == cond);
            if (hit is not null)
            {
                var tag = cond == condition ? "exact" : $"fallback({cond})";
                return new G2Lookup(Value(hit), hit.NLapsUsed, tag);
            }
        }
        var sameTC = Dto.G2TypByTrackCarCond.Where(
            r => r.TrackCanonical == track && r.Car == car).ToList();
        if (sameTC.Count > 0)
        {
            return new G2Lookup(sameTC.Average(Value),
                sameTC.Sum(r => r.NLapsUsed), "track_car_pooled");
        }
        var sameT = Dto.G2TypByTrackCarCond.Where(r => r.TrackCanonical == track).ToList();
        if (sameT.Count > 0)
        {
            return new G2Lookup(sameT.Average(Value),
                sameT.Sum(r => r.NLapsUsed), "track_pooled");
        }
        if (Dto.G2TypByTrackCarCond.Count > 0)
        {
            return new G2Lookup(Dto.G2TypByTrackCarCond.Average(Value), 0, "global");
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

    /// <summary>
    /// Typical out-lap (pit exit to the first start/finish crossing): rolling
    /// seconds and g², integrated first from the typed pit-exit temperature.
    /// Null when the artifact predates the table or has nothing for the track
    /// (the out-lap is then zero-length — pre-v0.26 behaviour). With a
    /// <paramref name="corner"/>, a v5 <c>outlap_q_by_corner</c> entry
    /// supplies that corner's heat input instead of <c>outlap_g2</c>.
    /// </summary>
    public OutlapLookup? LookupOutlap(string track, string car, string condition, string? corner = null)
    {
        var rows = Dto.OutlapTypByTrackCarCond;
        if (rows is null || rows.Count == 0) return null;
        double G2(OutlapEntryDto r) => PerCornerOr(r.OutlapQByCorner, corner, r.OutlapG2);

        foreach (var cond in ConditionChain(condition))
        {
            var hit = rows.FirstOrDefault(
                r => r.TrackCanonical == track && r.Car == car && r.Condition == cond);
            if (hit is not null)
            {
                var tag = cond == condition ? "exact" : $"fallback({cond})";
                return new OutlapLookup(hit.OutlapMovingS, G2(hit), hit.NLapsUsed, tag);
            }
        }
        var sameTC = rows.Where(r => r.TrackCanonical == track && r.Car == car).ToList();
        if (sameTC.Count > 0)
        {
            return new OutlapLookup(sameTC.Average(r => r.OutlapMovingS), sameTC.Average(G2),
                sameTC.Sum(r => r.NLapsUsed), "track_car_pooled");
        }
        var sameT = rows.Where(r => r.TrackCanonical == track).ToList();
        if (sameT.Count > 0)
        {
            return new OutlapLookup(sameT.Average(r => r.OutlapMovingS), sameT.Average(G2),
                sameT.Sum(r => r.NLapsUsed), "track_pooled");
        }
        return null;
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

public readonly record struct OutlapLookup(
    double MovingSeconds, double G2, int NLapsUsed, string Source);

public readonly record struct TauLookup(
    double ValueSeconds, double StderrSeconds, string SourceBucket, bool FromPrior);

public readonly record struct KLookup(
    double ValueKelvinPerG2, double StderrKelvinPerG2, int NSamples, string SourceBucket, bool FromPrior);

public readonly record struct G2Lookup(
    double Value, int NLapsUsed, string Source);

public readonly record struct LapTimeLookup(
    double ValueSeconds, int NLapsUsed, string Source);

public readonly record struct CornerDefaultsLookup(
    double HotTempC, double HotPressureBar, int NLapsUsed, string Source);

public readonly record struct G2PaceScale(
    double Scale, string Source);
