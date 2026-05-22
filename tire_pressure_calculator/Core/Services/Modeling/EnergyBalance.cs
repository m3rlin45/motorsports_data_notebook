using System;

namespace TirePressureCalculator.Services.Modeling;

/// <summary>
/// Pure physics ported verbatim from
/// <c>src/motorsports_data_notebook/tire_model/energy_balance.py</c>.
///
/// <para>
/// Single source of truth for both the prediction pipeline AND the
/// manual cold-pressure calculator (<see cref="ViewModels.TireCornerViewModel"/>).
/// Any change here MUST be mirrored in the Python module and exercised
/// by <c>CircuitPredictorTests</c>'s Python-parity fixture.
/// </para>
/// </summary>
public static class EnergyBalance
{
    // Atmospheric assumption — matches both the Python module and the
    // C# manual calculator that has used these constants since v1.
    public const double PAtmBar = 1.0;
    public const double TZeroCToK = 273.15;

    public static double TEffectiveC(double tAirC, double tRoadC, double wRoad)
    {
        if (wRoad < 0.0 || wRoad > 1.0)
            throw new ArgumentOutOfRangeException(nameof(wRoad), wRoad, "w_road must be in [0, 1]");
        return (1.0 - wRoad) * tAirC + wRoad * tRoadC;
    }

    public static double WarmupCurveC(
        double tSeconds,
        double tEffC,
        double kKelvinPerG2,
        double cTrack,
        double g2Typ,
        double tauSec)
    {
        if (tauSec <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(tauSec), tauSec, "tau_sec must be > 0");
        if (tSeconds < 0.0)
            throw new ArgumentOutOfRangeException(nameof(tSeconds), tSeconds, "t_seconds must be >= 0");
        double warmupFrac = 1.0 - Math.Exp(-tSeconds / tauSec);
        double deltaTInf = kKelvinPerG2 * cTrack * g2Typ;
        return tEffC + deltaTInf * warmupFrac;
    }

    /// <summary>
    /// Invert Gay-Lussac to compute cold pressure from target hot pressure +
    /// hot/cold temperatures. Matches the convention used by the Python
    /// predictor and by the manual calculator (gauge ↔ absolute via +1 bar).
    /// </summary>
    public static double GayLussacColdPressureBar(
        double targetHotPressureBar,
        double tHotC,
        double tColdC,
        double pAtmBar = PAtmBar)
    {
        double tColdK = tColdC + TZeroCToK;
        double tHotK = tHotC + TZeroCToK;
        if (tHotK <= 0.0 || tColdK <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(tHotC),
                $"Absolute temperatures must be positive (got T_hot_K={tHotK}, T_cold_K={tColdK})");
        double pHotAbs = targetHotPressureBar + pAtmBar;
        double pColdAbs = pHotAbs * (tColdK / tHotK);
        return pColdAbs - pAtmBar;
    }

    /// <summary>
    /// Track-surface temperature proxy from air temp + cloud cover.
    /// 0 % cloud cover ⇒ T_air + delta_sun_max_c; 100 % ⇒ T_air.
    /// </summary>
    public static double TRoadProxyC(
        double tAirC,
        double? cloudCoverPct,
        double sunFactor = 1.0,
        double deltaSunMaxC = 10.0)
    {
        if (cloudCoverPct is null) return tAirC;
        double clamped = Math.Max(0.0, Math.Min(100.0, cloudCoverPct.Value));
        return tAirC + deltaSunMaxC * (1.0 - clamped / 100.0) * sunFactor;
    }

    /// <summary>
    /// One step of the discretized ODE — not used at inference in v0 but
    /// kept available for tests that pin <see cref="WarmupCurveC"/> against
    /// its continuous-time integral.
    /// </summary>
    public static double WarmupRecurrenceStepC(
        double tCurrentC,
        double g2Current,
        double dtSeconds,
        double tEffC,
        double kKelvinPerG2,
        double cTrack,
        double tauSec)
    {
        if (tauSec <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(tauSec), tauSec, "tau_sec must be > 0");
        if (dtSeconds <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(dtSeconds), dtSeconds, "dt_seconds must be > 0");
        double decay = Math.Exp(-dtSeconds / tauSec);
        double growth = 1.0 - decay;
        return tEffC + (tCurrentC - tEffC) * decay + kKelvinPerG2 * cTrack * g2Current * growth;
    }
}
