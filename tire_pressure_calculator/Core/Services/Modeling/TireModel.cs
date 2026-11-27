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

    /// <summary>Where an entry row's numbers come from (its <c>source</c>,
    /// absent meaning logged): "sim" is a lap simulated on a committed
    /// track, whose road bank is processed from GSI's elevation tiles, so a
    /// prediction that reads one shows GSI's attribution. Mirrors
    /// predict.py.</summary>
    public const string Sim = "sim";
    public const string Logged = "logged";

    /// <summary>How a simulated pace curve answers the prediction's lap
    /// time (its <c>support</c>): answered either way, flagged unless "ok".
    /// Mirrors predict.py.</summary>
    public const string PaceOk = "ok";
    public const string PaceOutsideSupport = "outside_support";
    public const string PaceNoSolution = "no_solution";

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

    /// <summary>The cars the model's constants were fitted as (FJ, Inferno 86).</summary>
    public IReadOnlyList<string> FittedCars => Dto.TauSecByCarCornerCond
        .Select(r => r.Car).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();

    public bool IsFittedCar(string car) => FittedCars.Contains(car);

    /// <summary>
    /// The cars offered for selection: the fitted cars, with a pooled car
    /// whose raw cars have their own entries in the artifact (#383) listed
    /// as those raw cars (KK-SII, KK-F, KK-F-V2 in place of FJ). An artifact
    /// without raw-car rows offers the fitted cars, as before. Mirrors the
    /// web app's <c>availableCars</c>.
    /// </summary>
    public IReadOnlyList<string> AvailableCars
    {
        get
        {
            var fitted = FittedCars;
            var split = (Dto.G2TypByTrackRawCarCond ?? Array.Empty<G2EntryDto>())
                .Select(r => r.Car)
                .Where(raw => ResolveCar(raw) != raw && fitted.Contains(ResolveCar(raw)))
                .Select(ResolveCar)
                .ToHashSet();
            var raws = (Dto.CarAliases ?? new Dictionary<string, string>())
                .Where(kv => kv.Value != kv.Key && split.Contains(kv.Value))
                .Select(kv => kv.Key);
            return fitted.Where(c => !split.Contains(c)).Concat(raws)
                .OrderBy(s => s, StringComparer.Ordinal).ToList();
        }
    }

    /// <summary>
    /// The car to select for a saved one: kept where it is offered; a
    /// fitted (pooled) name saved before its raw cars were listed keeps
    /// working on the pooled entries; a raw name the artifact does not list
    /// resolves to its pooled car, as before. Null for a car the model does
    /// not know. Mirrors the web app's <c>selectableCar</c>.
    /// </summary>
    public string? SelectableCar(string? saved)
    {
        if (saved is null) return null;
        var offered = AvailableCars;
        if (offered.Contains(saved) || IsFittedCar(saved)) return saved;
        var pooled = ResolveCar(saved);
        return offered.Contains(pooled) || IsFittedCar(pooled) ? pooled : null;
    }

    /// <summary>Resolve an alias-pooled car name (e.g. KK-F / KK-SII → FJ)
    /// onto the label the model was fitted with. Unknown cars pass through.</summary>
    public string ResolveCar(string car) =>
        Dto.CarAliases is { } aliases && aliases.TryGetValue(car, out var pooled) ? pooled : car;

    /// <summary>
    /// The row of an entry table that answers (track, car, condition) at one
    /// step of the condition chain: a raw car's own row in the raw-car table
    /// (<paramref name="rawRows"/>, #383) where it has one, else the fitted
    /// car's in <paramref name="rows"/>. <paramref name="car"/> may be either
    /// name. A simulated row sits beside the logged one and answers only its
    /// own exact condition (schema §9), chosen by its <c>source</c> (absent:
    /// logged), never by its place in the table: at the
    /// <paramref name="exact"/> step the order is raw sim, raw logged, pooled
    /// sim, pooled logged; at any other step (a condition fallback) only
    /// logged rows count, the raw car's before the pooled car's. Mirrors
    /// predict.py's <c>_entry_row</c>.
    /// </summary>
    private T? EntryRow<T>(IReadOnlyList<T>? rows, IReadOnlyList<T>? rawRows,
        string track, string car, string condition, bool exact) where T : class, IEntryRow
    {
        var fit = ResolveCar(car);
        var tables = new List<(IReadOnlyList<T>? Rows, string Who)>();
        if (fit != car) tables.Add((rawRows, car));
        tables.Add((rows, fit));
        var kinds = exact ? new[] { true, false } : new[] { false };
        foreach (var (table, who) in tables)
        {
            var atKey = table?.Where(r => r.TrackCanonical == track && r.Car == who && r.Condition == condition)
                .ToList() ?? new List<T>();
            foreach (var sim in kinds)
            {
                var hit = atKey.FirstOrDefault(r => IsSim(r) == sim);
                if (hit is not null) return hit;
            }
        }
        return null;
    }

    /// <summary>Every condition's logged row at <paramref name="track"/> for
    /// the car whose entries answer there: the raw car's own where it has
    /// any, else the fitted car's (the track_car_pooled fallback; an
    /// average, so a simulated row is not counted).</summary>
    private List<T> EntryRowsAtTrack<T>(IReadOnlyList<T>? rows, IReadOnlyList<T>? rawRows,
        string track, string car) where T : class, IEntryRow
    {
        var fit = ResolveCar(car);
        if (fit != car && rawRows is not null)
        {
            var own = rawRows.Where(r => r.TrackCanonical == track && r.Car == car && !IsSim(r)).ToList();
            if (own.Count > 0) return own;
        }
        return rows?.Where(r => r.TrackCanonical == track && r.Car == fit && !IsSim(r)).ToList()
            ?? new List<T>();
    }

    private static bool IsSim(IEntryRow r) => (r.Source ?? Logged) == Sim;

    /// <summary>The steps of <see cref="AnsweringRows"/> that answer with an
    /// average of rows.</summary>
    private static bool IsAveraged(string step) =>
        step is "track_car_pooled" or "track_pooled" or "global";

    /// <summary>
    /// The rows of an entry table that answer (track, car, condition), and
    /// the step that found them: one row by the condition chain ("exact",
    /// "fallback(&lt;cond&gt;)"; a raw car's own first; a simulated row only at
    /// the exact condition), else every
    /// condition's row at the track for the car ("track_car_pooled"), else
    /// the pooled rows at the track ("track_pooled"), else, with
    /// <paramref name="globalStep"/>, every pooled row ("global"; none:
    /// "global" with no rows). The averaged steps read logged rows only (the
    /// lead's ruling of 2026-11-27, so that a car with no simulated entry of
    /// its own reads what it read before), and a step the filter leaves empty
    /// is skipped. Every lookup walks this one chain, and a prediction's
    /// provenance is read from the rows it found. Mirrors predict.py's
    /// <c>_answering_rows</c>.
    /// </summary>
    internal (List<T> Rows, string Step) AnsweringRows<T>(IReadOnlyList<T>? rows,
        IReadOnlyList<T>? rawRows, string track, string car, string condition, bool globalStep)
        where T : class, IEntryRow
    {
        foreach (var cond in ConditionChain(condition))
        {
            bool exact = cond == condition;
            var hit = EntryRow(rows, rawRows, track, car, cond, exact);
            if (hit is not null) return (new List<T> { hit }, exact ? "exact" : $"fallback({cond})");
        }
        var sameTC = EntryRowsAtTrack(rows, rawRows, track, car);
        if (sameTC.Count > 0) return (sameTC, "track_car_pooled");
        var logged = rows?.Where(r => !IsSim(r)).ToList() ?? new List<T>();
        var sameT = logged.Where(r => r.TrackCanonical == track).ToList();
        if (sameT.Count > 0) return (sameT, "track_pooled");
        return (globalStep ? logged : new List<T>(), "global");
    }

    /// <summary>"sim" if any of the rows is simulated, else "logged".</summary>
    public static string RowProvenance<T>(IEnumerable<T> rows) where T : IEntryRow =>
        rows.Any(r => IsSim(r)) ? Sim : Logged;

    internal (List<G2EntryDto> Rows, string Step) HeatRows(string track, string car, string condition) =>
        AnsweringRows(Dto.G2TypByTrackCarCond, Dto.G2TypByTrackRawCarCond, track, car, condition, globalStep: true);

    internal (List<LapTimeEntryDto> Rows, string Step) LapTimeRows(string track, string car, string condition) =>
        AnsweringRows(Dto.LapTimeTypByTrackCarCond, Dto.LapTimeTypByTrackRawCarCond, track, car, condition, globalStep: false);

    internal (List<OutlapEntryDto> Rows, string Step) OutlapRows(string track, string car, string condition) =>
        AnsweringRows(Dto.OutlapTypByTrackCarCond, Dto.OutlapTypByTrackRawCarCond, track, car, condition, globalStep: false);

    /// <summary>
    /// How the simulated pace curve a prediction reads answers
    /// <paramref name="lapTimeS"/> (the target, else the typical lap time):
    /// "no_solution" under its fastest knot (answered there, the clamp),
    /// "outside_support" outside its <c>support</c> (the lap times of the
    /// car's logged range of use at the layout) or past its slowest knot,
    /// else "ok"; null where the row the pace scale reads is not simulated
    /// (or is an average of rows). Display only. Mirrors predict.py's
    /// <c>pace_status</c>.
    /// </summary>
    public string? PaceStatus(string track, string car, string condition, double lapTimeS)
    {
        var (rows, step) = HeatRows(track, car, condition);
        if (IsAveraged(step) || RowProvenance(rows) != Sim) return null;
        if (rows[0].G2VsLapTime is not { } curve) return null;
        var xs = curve.LapTimeS;
        if (!(lapTimeS >= xs[0])) return PaceNoSolution;
        if (curve.Support is not { LapTimeMinS: double lo, LapTimeMaxS: double hi }
            || lapTimeS > xs[xs.Count - 1] || !(lo <= lapTimeS && lapTimeS <= hi))
            return PaceOutsideSupport;
        return PaceOk;
    }

    /// <summary>A pooled car selected where its raw cars are offered in its
    /// place (a saved "FJ" from before #383): it reads the pooled, logged
    /// entries, and the UI hints to pick the raw car for its own. Mirrors
    /// the web app's <c>isPooledSelection</c>.</summary>
    public bool IsPooledSelection(string? car) =>
        car is not null && IsFittedCar(car) && !AvailableCars.Contains(car);

    // Every track with an entry (a typical heat input), a raw car's own
    // included (a track only the simulation covers for it).
    public IReadOnlyList<string> AvailableTracks => Dto.G2TypByTrackCarCond
        .Concat(Dto.G2TypByTrackRawCarCond ?? Array.Empty<G2EntryDto>())
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

    // The constants (tau, K, compound K, corner defaults, hot-pressure
    // targets) are the fitted car's: a raw name resolves through CarAliases.
    public TauLookup LookupTau(string car, string corner, string condition)
    {
        car = ResolveCar(car);
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
        car = ResolveCar(car);
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
    public IReadOnlyList<string> AvailableCompounds(string car)
    {
        car = ResolveCar(car);
        return (Dto.KByCarCompoundCornerCond ?? Array.Empty<KCompoundEntryDto>())
            .Where(r => r.Car == car).Select(r => r.Compound)
            .Distinct().OrderBy(s => s).ToList();
    }

    /// <summary>Compound-specific K via the condition chain, or null when
    /// the artifact has no fitted bucket (caller falls back to pooled K).</summary>
    public KLookup? LookupCompoundK(string car, string compound, string corner, string condition)
    {
        car = ResolveCar(car);
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
        car = ResolveCar(car);
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

    /// <summary>Team hot-pressure target for (car, compound) in bar gauge,
    /// or null when the artifact has none for that pair (the caller falls
    /// back to the measured median).</summary>
    public double? LookupTargetHotPressure(string car, string? compound)
    {
        if (compound is null) return null;
        car = ResolveCar(car);
        var rows = Dto.TargetHotPressureByCarCompound ?? Array.Empty<TargetHotPressureEntryDto>();
        return rows.FirstOrDefault(r => r.Car == car && r.Compound == compound)?.TargetHotPressureBar;
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

        var (rows, step) = HeatRows(track, car, condition);
        if (rows.Count == 0) return new G2Lookup(0.7, 0, "global");
        if (!IsAveraged(step)) return new G2Lookup(Value(rows[0]), rows[0].NLapsUsed, step);
        return new G2Lookup(rows.Average(Value), step == "global" ? 0 : rows.Sum(r => r.NLapsUsed), step);
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

    /// <summary>The pace curve of the row <see cref="LookupG2"/> reads (a
    /// condition-chain hit; an average of rows has none).</summary>
    private G2CurveDto? LookupG2PaceCurve(string track, string car, string condition)
    {
        var (rows, step) = HeatRows(track, car, condition);
        return IsAveraged(step) || rows.Count == 0 ? null : rows[0].G2VsLapTime;
    }

    /// <summary>
    /// Multiplier on the typical heat input for a target lap time; mirrors
    /// the Python predictor's <c>_g2_pace_scale</c>. Preferred: ratio along
    /// the bucket's simulated-lap curve — the corner's own series when
    /// <paramref name="corner"/> is given and the curve carries
    /// <c>g2_by_corner</c>, else the corner mean (anchored at lap_time_typ
    /// so target == typical scales by exactly 1). Fallback: the physics
    /// exponent. Clamped either way.
    /// </summary>
    public G2PaceScale ComputeG2PaceScale(
        string track, string car, string condition,
        double lapTimeTypS, double targetLapTimeS, string? corner = null)
    {
        double clampMin = Dto.G2LapTimeModel?.MultiplierClamp?.Min ?? 0.4;
        double clampMax = Dto.G2LapTimeModel?.MultiplierClamp?.Max ?? 2.5;

        var curve = LookupG2PaceCurve(track, car, condition);
        if (curve is not null)
        {
            IReadOnlyList<double> series = curve.G2;
            if (corner is not null && curve.G2ByCorner is not null
                && curve.G2ByCorner.TryGetValue(corner, out var own))
                series = own;
            double reference = InterpClamped(lapTimeTypS, curve.LapTimeS, series);
            if (reference > 0)
            {
                double curveScale = InterpClamped(targetLapTimeS, curve.LapTimeS, series) / reference;
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
        double G2(OutlapEntryDto r) => PerCornerOr(r.OutlapQByCorner, corner, r.OutlapG2);

        var (rows, step) = OutlapRows(track, car, condition);
        if (rows.Count == 0) return null;
        if (!IsAveraged(step))
            return new OutlapLookup(rows[0].OutlapMovingS, G2(rows[0]), rows[0].NLapsUsed, step);
        return new OutlapLookup(rows.Average(r => r.OutlapMovingS), rows.Average(G2),
            rows.Sum(r => r.NLapsUsed), step);
    }

    public LapTimeLookup LookupLapTime(string track, string car, string condition)
    {
        var (rows, step) = LapTimeRows(track, car, condition);
        if (rows.Count == 0) return new LapTimeLookup(90.0, 0, "global");
        if (!IsAveraged(step)) return new LapTimeLookup(rows[0].LapTimeTypS, rows[0].NLapsUsed, step);
        return new LapTimeLookup(rows.Average(r => r.LapTimeTypS), rows.Sum(r => r.NLapsUsed), step);
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
