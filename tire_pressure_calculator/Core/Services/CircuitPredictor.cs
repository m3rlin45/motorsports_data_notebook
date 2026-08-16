using System;
using TirePressureCalculator.Services.Modeling;

namespace TirePressureCalculator.Services;

/// <summary>
/// Per-corner cold-pressure predictor. C# port of
/// <c>predict_cold_pressure(...)</c> from
/// <c>src/motorsports_data_notebook/tire_model/predict.py</c>.
///
/// One <see cref="CircuitPredictor"/> wraps one <see cref="TireModel"/>;
/// callers reuse it across the four corners and across input changes.
/// </summary>
public sealed class CircuitPredictor
{
    private readonly TireModel _model;

    public CircuitPredictor(TireModel model) => _model = model;

    public TireModel Model => _model;

    /// <summary>
    /// Predict a single corner. Returns the recommended cold pressure (bar
    /// gauge) plus the intermediate quantities the UI displays.
    /// </summary>
    public CornerPrediction Predict(
        string track,
        string car,
        string condition,
        int lapWithinStint,
        double ambientTempC,
        double? trackTempC,
        double? cloudCoverPct,
        string corner,
        double targetHotPressureBar,
        double? coldTireTempC = null,
        double? targetLapTimeS = null)
    {
        ArgumentNullException.ThrowIfNull(track);
        ArgumentNullException.ThrowIfNull(car);
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(corner);

        var cond = condition.ToLowerInvariant();
        if (cond is not "dry" and not "damp" and not "wet")
            throw new ArgumentException(
                $"track_condition must be dry/damp/wet; got '{condition}'", nameof(condition));

        var k = _model.LookupK(car, corner, cond);
        var tau = _model.LookupTau(car, corner, cond);
        var c = _model.LookupCTrack(track);
        var g2 = _model.LookupG2(track, car, cond);
        var lap = _model.LookupLapTime(track, car, cond);

        // T_road: user-supplied → sun-cover proxy → fall back to T_air.
        double tRoadC = trackTempC ?? EnergyBalance.TRoadProxyC(
            tAirC: ambientTempC,
            cloudCoverPct: cloudCoverPct,
            sunFactor: _model.SunFactorDefault,
            deltaSunMaxC: _model.DeltaSunMaxC);
        double tEffC = EnergyBalance.TEffectiveC(ambientTempC, tRoadC, _model.WRoad);

        double tColdC = coldTireTempC ?? ambientTempC;

        // Target-lap-time feature: pace sets both time-on-track and tire energy.
        double g2Scale = 1.0;
        string? g2PaceSource = null;
        double lapTimeForClockS = lap.ValueSeconds;
        double g2Value = g2.Value;
        if (targetLapTimeS is double target)
        {
            if (target <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(targetLapTimeS), target, "target lap time must be > 0");
            var pace = _model.ComputeG2PaceScale(track, car, cond, lap.ValueSeconds, target);
            g2Scale = pace.Scale;
            g2PaceSource = pace.Source;
            g2Value *= g2Scale;
            lapTimeForClockS = target;
        }

        double tAtLapNs = (double)lapWithinStint * lapTimeForClockS;
        double warmupFrac = tau.ValueSeconds > 0
            ? 1.0 - Math.Exp(-tAtLapNs / tau.ValueSeconds)
            : 0.0;
        double deltaTInf = k.ValueKelvinPerG2 * c.Value * g2Value;
        double tHotC = EnergyBalance.WarmupCurveC(
            tSeconds: tAtLapNs,
            tEffC: tEffC,
            kKelvinPerG2: k.ValueKelvinPerG2,
            cTrack: c.Value,
            g2Typ: g2Value,
            tauSec: tau.ValueSeconds);

        double cold = EnergyBalance.GayLussacColdPressureBar(
            targetHotPressureBar: targetHotPressureBar,
            tHotC: tHotC,
            tColdC: tColdC,
            pAtmBar: _model.PAtmBar);

        return new CornerPrediction(
            Corner: corner,
            ColdPressureBar: cold,
            PredictedHotTempC: tHotC,
            TargetHotPressureBar: targetHotPressureBar,
            KKelvinPerG2: k.ValueKelvinPerG2,
            TauSec: tau.ValueSeconds,
            CTrack: c.Value,
            G2Typ: g2Value,
            LapTimeTypS: lap.ValueSeconds,
            TAtLapNs: tAtLapNs,
            WarmupFrac: warmupFrac,
            DeltaTInfKelvin: deltaTInf,
            TEffC: tEffC,
            TAirC: ambientTempC,
            TRoadC: tRoadC,
            TColdC: tColdC,
            KSourceBucket: k.SourceBucket,
            KFromPrior: k.FromPrior,
            KNSamples: k.NSamples,
            TargetLapTimeS: targetLapTimeS,
            G2Scale: g2Scale,
            G2PaceSource: g2PaceSource);
    }
}

public sealed record CornerPrediction(
    string Corner,
    double ColdPressureBar,
    double PredictedHotTempC,
    double TargetHotPressureBar,
    double KKelvinPerG2,
    double TauSec,
    double CTrack,
    double G2Typ,
    double LapTimeTypS,
    double TAtLapNs,
    double WarmupFrac,
    double DeltaTInfKelvin,
    double TEffC,
    double TAirC,
    double TRoadC,
    double TColdC,
    string KSourceBucket,
    bool KFromPrior,
    int KNSamples,
    double? TargetLapTimeS = null,
    double G2Scale = 1.0,
    string? G2PaceSource = null);
