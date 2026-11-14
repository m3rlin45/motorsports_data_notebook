namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Throttle acceptance for one corner exit; port of the <c>find_throttle_acceptance</c> result dict.</summary>
/// <param name="ThrottleAcceptancePct">Lateral G at sustained full throttle as % of the corner's peak.</param>
/// <param name="LateralGAtThrottle">Smoothed |lateral G| when full throttle was reached.</param>
/// <param name="PeakLateralG">Peak smoothed |lateral G| in the corner.</param>
/// <param name="FullThrottleDist">Distance where sustained full throttle began, metres.</param>
public sealed record ThrottleAcceptance(double ThrottleAcceptancePct, double LateralGAtThrottle, double PeakLateralG, double FullThrottleDist);

/// <summary>Driver behaviour metrics; port of <c>driver_analysis.py</c>.</summary>
public static class DriverAnalysis
{
    /// <summary>
    /// <c>prepare_throttle_acceptance</c>: smoothed |lateral G| (box convolution,
    /// numpy "same") and the effective full-throttle threshold, scaled down for
    /// sensors whose maximum sits between 90 and 100 %.
    /// </summary>
    public static (double[] SmoothedLateralG, double EffectiveThreshold) PrepareThrottleAcceptance(
        ReadOnlySpan<double> throttle, ReadOnlySpan<double> lateralG, double? throttleThreshold = null, int smoothingWindow = 25)
    {
        var threshold = throttleThreshold ?? 0.98 * Numerics.InferChannelScale(throttle);
        var absLateral = new double[lateralG.Length];
        for (var i = 0; i < absLateral.Length; i++)
        {
            absLateral[i] = Math.Abs(lateralG[i]);
        }

        var kernel = new double[smoothingWindow];
        Array.Fill(kernel, 1.0 / smoothingWindow);
        var smoothed = Numerics.ConvolveSame(absLateral, kernel);

        var maxThrottle = Numerics.Max(throttle);
        var effective = maxThrottle >= 90.0 && maxThrottle < 100.0 ? maxThrottle * (threshold / 100.0) : threshold;
        return (smoothed, effective);
    }

    /// <summary>
    /// <c>find_throttle_acceptance_from_arrays</c>: in the exit zone (apex to
    /// corner end), the first sample at or above the threshold that stays
    /// there for <paramref name="sustainTimeMs"/>; null when the corner has no
    /// lateral load or full throttle is never sustained.
    /// </summary>
    public static ThrottleAcceptance? FindThrottleAcceptance(
        ReadOnlySpan<double> distance,
        ReadOnlySpan<long> timecodes,
        ReadOnlySpan<double> throttle,
        ReadOnlySpan<double> lateralG,
        Corner corner,
        double[]? smoothedLateralG = null,
        double? effectiveThreshold = null,
        double? throttleThreshold = null,
        double sustainTimeMs = 500.0,
        int smoothingWindow = 25)
    {
        if (smoothedLateralG is null || effectiveThreshold is null)
        {
            (smoothedLateralG, effectiveThreshold) = PrepareThrottleAcceptance(throttle, lateralG, throttleThreshold, smoothingWindow);
        }

        var cSi = Numerics.SearchSortedLeft(distance, corner.StartDist);
        var cEi = Numerics.SearchSortedRight(distance, corner.EndDist);
        if (cSi >= cEi)
        {
            return null;
        }

        var peak = Numerics.Max(smoothedLateralG.AsSpan(cSi, cEi - cSi));
        if (peak < 0.1)
        {
            return null;
        }

        var aSi = Numerics.SearchSortedLeft(distance, corner.ApexDist);
        if (aSi >= cEi)
        {
            return null;
        }

        var n = cEi - aSi;
        for (var j = 0; j < n; j++)
        {
            if (throttle[aSi + j] < effectiveThreshold.Value)
            {
                continue;
            }

            var endTime = timecodes[aSi + j] + sustainTimeMs;
            var sustained = true;
            for (var k = j + 1; k < n; k++)
            {
                if (timecodes[aSi + k] > endTime)
                {
                    break;
                }

                if (throttle[aSi + k] < effectiveThreshold.Value)
                {
                    sustained = false;
                    break;
                }
            }

            if (sustained)
            {
                var lateralAt = smoothedLateralG[aSi + j];
                return new ThrottleAcceptance(lateralAt / peak * 100, lateralAt, peak, distance[aSi + j]);
            }
        }

        return null;
    }
}
