using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Velocity thresholds (mm/s, absolute) for categorising suspension motion.</summary>
public sealed record VelocityRanges(double Friction = 5.0, double Slow = 25.0, double Fast = 200.0);

/// <summary>Velocity histogram and statistics for one corner; port of <c>CornerVelocityData</c>.</summary>
public sealed record CornerVelocityData(
    string CornerName,
    double[] Velocity,
    double[] BinEdges,
    double[] BinCenters,
    double[] Histogram,
    double Skew,
    double Kurtosis,
    double Mean,
    double Std,
    double PctFriction,
    double PctSlowBump,
    double PctSlowRebound,
    double PctFastBump,
    double PctFastRebound,
    double PctCurb,
    double PctZeroBin)
{
    /// <summary>Bump share of all slow+fast motion, percent (50 when there is none); the symmetry table's "Bump/Total %".</summary>
    public double BumpOfTotalPct
    {
        get
        {
            var bump = PctSlowBump + PctFastBump;
            var rebound = PctSlowRebound + PctFastRebound;
            return bump + rebound > 0 ? bump / (bump + rebound) * 100 : 50.0;
        }
    }
}

/// <summary>All four corners; port of <c>VelocityHistogramResult</c>.</summary>
public sealed record VelocityHistogramResult(
    CornerVelocityData FrontLeft,
    CornerVelocityData FrontRight,
    CornerVelocityData RearLeft,
    CornerVelocityData RearRight,
    VelocityRanges VelocityRanges)
{
    /// <summary>FL, FR, RL, RR.</summary>
    public IEnumerable<CornerVelocityData> Corners => new[] { FrontLeft, FrontRight, RearLeft, RearRight };

    /// <summary>
    /// <c>format_comparison_table</c>: the per-range percentage differences
    /// between the left (FL+RL) and right (FR+RR) halves and between the
    /// front (FL+FR) and rear (RL+RR), each as group averages.
    /// </summary>
    public (GroupDifference LeftVsRight, GroupDifference FrontVsRear) GroupDifferences()
    {
        static GroupDifference Diff(CornerVelocityData a1, CornerVelocityData a2, CornerVelocityData b1, CornerVelocityData b2) => new(
            (a1.PctZeroBin + a2.PctZeroBin) / 2 - (b1.PctZeroBin + b2.PctZeroBin) / 2,
            (a1.PctFriction + a2.PctFriction) / 2 - (b1.PctFriction + b2.PctFriction) / 2,
            (a1.PctSlowBump + a2.PctSlowBump) / 2 - (b1.PctSlowBump + b2.PctSlowBump) / 2,
            (a1.PctSlowRebound + a2.PctSlowRebound) / 2 - (b1.PctSlowRebound + b2.PctSlowRebound) / 2,
            (a1.PctFastBump + a2.PctFastBump) / 2 - (b1.PctFastBump + b2.PctFastBump) / 2,
            (a1.PctFastRebound + a2.PctFastRebound) / 2 - (b1.PctFastRebound + b2.PctFastRebound) / 2,
            (a1.PctCurb + a2.PctCurb) / 2 - (b1.PctCurb + b2.PctCurb) / 2);

        return (Diff(FrontLeft, RearLeft, FrontRight, RearRight), Diff(FrontLeft, FrontRight, RearLeft, RearRight));
    }
}

/// <summary>Percentage-point differences between two wheel groups per velocity range.</summary>
public sealed record GroupDifference(double ZeroBin, double Friction, double SlowBump, double SlowRebound, double FastBump, double FastRebound, double Curb);

/// <summary>Suspension velocity histogram analysis; port of <c>suspension.py</c>.</summary>
public static class Suspension
{
    /// <summary>Canonical keys of the four shock channels, FL FR RL RR.</summary>
    public static readonly string[] ShockKeys = { "shock_fl", "shock_fr", "shock_rl", "shock_rr" };

    private static readonly string[] CornerLabels = { "FL", "FR", "RL", "RR" };

    /// <summary>
    /// <c>compute_shock_velocity</c>: d(displacement)/dt in mm/s from
    /// millisecond timecodes (numpy.gradient with coordinates), then an
    /// edge-padded moving average of <paramref name="smoothingWindow"/> samples.
    /// </summary>
    public static double[] ComputeShockVelocity(ReadOnlySpan<double> displacement, ReadOnlySpan<long> timecodes, int smoothingWindow = 5)
    {
        var seconds = new double[timecodes.Length];
        for (var i = 0; i < seconds.Length; i++)
        {
            seconds[i] = timecodes[i] / 1000.0;
        }

        var velocity = Numerics.Gradient(displacement, seconds);
        if (smoothingWindow > 1 && velocity.Length >= smoothingWindow)
        {
            velocity = EdgePaddedMovingAverage(velocity, smoothingWindow);
        }

        return velocity;
    }

    /// <summary><c>np.pad(mode="edge")</c> by window/2 each side, then <c>np.convolve(mode="valid")</c> with a box kernel.</summary>
    private static double[] EdgePaddedMovingAverage(double[] values, int window)
    {
        var pad = window / 2;
        var n = values.Length;
        var padded = new double[n + 2 * pad];
        for (var i = 0; i < padded.Length; i++)
        {
            padded[i] = values[Math.Clamp(i - pad, 0, n - 1)];
        }

        var weight = 1.0 / window;
        var outLength = padded.Length - window + 1;
        var result = new double[outLength];
        for (var i = 0; i < outLength; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < window; j++)
            {
                sum += padded[i + j] * weight;
            }

            result[i] = sum;
        }

        return result;
    }

    /// <summary><c>compute_wheel_velocity</c>: pot velocity / pot-per-wheel ratio.</summary>
    public static double[] ComputeWheelVelocity(ReadOnlySpan<double> shockVelocity, double motionRatio)
    {
        var result = new double[shockVelocity.Length];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = shockVelocity[i] / motionRatio;
        }

        return result;
    }

    /// <summary>
    /// <c>compute_velocity_histogram</c>: zero-centred bins of
    /// <paramref name="binSize"/> out to ±<paramref name="maxVelocity"/>,
    /// as percentages of all samples (velocities clipped into range).
    /// </summary>
    public static (double[] Histogram, double[] BinEdges, double[] BinCenters) ComputeVelocityHistogram(
        ReadOnlySpan<double> velocity, double binSize = 5.0, double maxVelocity = 300.0)
    {
        var halfBin = binSize / 2;
        var positive = Numerics.ARange(halfBin, maxVelocity + binSize, binSize);
        var edges = new double[positive.Length * 2];
        for (var i = 0; i < positive.Length; i++)
        {
            edges[i] = -positive[positive.Length - 1 - i];
            edges[positive.Length + i] = positive[i];
        }

        var clipped = new double[velocity.Length];
        for (var i = 0; i < clipped.Length; i++)
        {
            clipped[i] = Math.Clamp(velocity[i], -maxVelocity, maxVelocity);
        }

        var counts = Numerics.Histogram(clipped, edges);
        var histogram = new double[counts.Length];
        for (var i = 0; i < counts.Length; i++)
        {
            histogram[i] = velocity.Length > 0 ? (double)counts[i] / velocity.Length * 100.0 : counts[i];
        }

        var centers = new double[edges.Length - 1];
        for (var i = 0; i < centers.Length; i++)
        {
            centers[i] = (edges[i] + edges[i + 1]) / 2;
        }

        return (histogram, edges, centers);
    }

    /// <summary><c>compute_velocity_stats</c>: moments and time shares per velocity range.</summary>
    public static (double Skew, double Kurtosis, double Mean, double Std, double PctFriction, double PctSlowBump, double PctSlowRebound, double PctFastBump, double PctFastRebound, double PctCurb)
        ComputeVelocityStats(ReadOnlySpan<double> velocity, VelocityRanges ranges)
    {
        var n = velocity.Length;
        if (n == 0)
        {
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        var mean = Numerics.Mean(velocity);
        var std = Numerics.Std(velocity);
        double skew = 0, kurtosis = 0;
        if (std > 0)
        {
            var cubes = new double[n];
            var fourths = new double[n];
            for (var i = 0; i < n; i++)
            {
                var z = (velocity[i] - mean) / std;
                cubes[i] = Math.Pow(z, 3);
                fourths[i] = Math.Pow(z, 4);
            }

            skew = Numerics.Mean(cubes);
            kurtosis = Numerics.Mean(fourths) - 3;
        }

        long friction = 0, slowBump = 0, slowRebound = 0, fastBump = 0, fastRebound = 0, curb = 0;
        foreach (var v in velocity)
        {
            var a = Math.Abs(v);
            if (a < ranges.Friction)
            {
                friction++;
            }
            else if (a < ranges.Slow)
            {
                if (v > 0) slowBump++;
                else if (v < 0) slowRebound++;
            }
            else if (a < ranges.Fast)
            {
                if (v > 0) fastBump++;
                else if (v < 0) fastRebound++;
            }
            else
            {
                curb++;
            }
        }

        static double Pct(long count, int n) => (double)count / n * 100;
        return (skew, kurtosis, mean, std, Pct(friction, n), Pct(slowBump, n), Pct(slowRebound, n), Pct(fastBump, n), Pct(fastRebound, n), Pct(curb, n));
    }

    /// <summary><c>_build_corner_data</c>: histogram, zero-bin share and stats for one corner's wheel velocity (mm/s).</summary>
    public static CornerVelocityData BuildCornerData(double[] velocityMms, string cornerName, double binSize, double maxVelocity, VelocityRanges ranges)
    {
        var (histogram, edges, centers) = ComputeVelocityHistogram(velocityMms, binSize, maxVelocity);
        var absCenters = centers.Select(Math.Abs).ToArray();
        var zeroBin = Numerics.ArgMin(absCenters);
        var pctZeroBin = histogram.Length > 0 ? histogram[zeroBin] : 0.0;
        var s = ComputeVelocityStats(velocityMms, ranges);
        return new CornerVelocityData(cornerName, velocityMms, edges, centers, histogram, s.Skew, s.Kurtosis, s.Mean, s.Std, s.PctFriction, s.PctSlowBump, s.PctSlowRebound, s.PctFastBump, s.PctFastRebound, s.PctCurb, pctZeroBin);
    }

    /// <summary>Whether a shock channel already holds velocity (unit containing "/s", as iRacing's do) rather than displacement.</summary>
    public static bool ChannelIsVelocity(LogFile log, string channelName) =>
        log.TryGetChannel(channelName, out var channel) && channel.Units.Contains("/s");

    /// <summary>
    /// <c>analyze_suspension_velocity</c> on a lap-filtered session: the four
    /// shock channels resampled onto the FL channel, pot velocity derived
    /// and divided by each corner's pot/wheel ratio (or taken as-is × 1000
    /// for velocity channels).
    /// </summary>
    public static VelocityHistogramResult AnalyzeSuspensionVelocity(
        LogFile lap,
        IReadOnlyDictionary<string, string> channelNames,
        MotionRatios? motionRatios = null,
        VelocityRanges? velocityRanges = null,
        int smoothingWindow = 5,
        double binSize = 10.0,
        double maxVelocity = 300.0)
    {
        var ranges = velocityRanges ?? new VelocityRanges();
        var velocities = LapWheelVelocities(lap, channelNames, motionRatios ?? MotionRatios.Toyota86Zn6, smoothingWindow, ChannelIsVelocity(lap, channelNames["shock_fl"]));
        return Build(velocities, binSize, maxVelocity, ranges);
    }

    /// <summary>
    /// <c>analyze_suspension_velocity_multi_lap</c>: wheel velocities of every
    /// lap concatenated before the histogram, so longer laps weigh more.
    /// Laps that cannot be processed are skipped, as in Python.
    /// </summary>
    public static VelocityHistogramResult AnalyzeSuspensionVelocityMultiLap(
        LogFile log,
        IReadOnlyList<int> lapNumbers,
        IReadOnlyDictionary<string, string> channelNames,
        MotionRatios? motionRatios = null,
        VelocityRanges? velocityRanges = null,
        int smoothingWindow = 5,
        double binSize = 10.0,
        double maxVelocity = 300.0)
    {
        if (lapNumbers.Count == 0)
        {
            throw new ArgumentException("lap_numbers cannot be empty", nameof(lapNumbers));
        }

        var ranges = velocityRanges ?? new VelocityRanges();
        var isVelocity = ChannelIsVelocity(log, channelNames["shock_fl"]);
        var all = CornerLabels.ToDictionary(c => c, _ => new List<double[]>());
        foreach (var lapNum in lapNumbers)
        {
            double[][] velocities;
            try
            {
                velocities = LapWheelVelocities(log.FilterByLap(lapNum), channelNames, motionRatios ?? MotionRatios.Toyota86Zn6, smoothingWindow, isVelocity);
            }
            catch (Exception e) when (e is KeyNotFoundException or ArgumentException or InvalidOperationException)
            {
                continue;
            }

            for (var c = 0; c < 4; c++)
            {
                all[CornerLabels[c]].Add(velocities[c]);
            }
        }

        var combined = CornerLabels.Select(c => all[c].SelectMany(v => v).ToArray()).ToArray();
        return Build(combined, binSize, maxVelocity, ranges);
    }

    private static VelocityHistogramResult Build(double[][] velocities, double binSize, double maxVelocity, VelocityRanges ranges)
    {
        var corners = new CornerVelocityData[4];
        for (var c = 0; c < 4; c++)
        {
            corners[c] = BuildCornerData(velocities[c], CornerLabels[c], binSize, maxVelocity, ranges);
        }

        return new VelocityHistogramResult(corners[0], corners[1], corners[2], corners[3], ranges);
    }

    /// <summary>Wheel velocities (mm/s) of FL, FR, RL, RR for one lap-filtered session.</summary>
    private static double[][] LapWheelVelocities(LogFile lap, IReadOnlyDictionary<string, string> channelNames, MotionRatios ratios, int smoothingWindow, bool isVelocity)
    {
        var names = ShockKeys.Select(k => channelNames[k]).ToArray();
        var aligned = lap.SelectChannels(names).ResampleToChannel(names[0]);
        var mr = new[] { ratios.FrontLeft, ratios.FrontRight, ratios.RearLeft, ratios.RearRight };
        var result = new double[4][];
        var timecodes = aligned[names[0]].Timecodes;
        for (var c = 0; c < 4; c++)
        {
            var values = aligned[names[c]].Values;
            if (isVelocity)
            {
                result[c] = values.Select(v => v * 1000.0).ToArray();
            }
            else
            {
                result[c] = ComputeWheelVelocity(ComputeShockVelocity(values, timecodes, smoothingWindow), mr[c]);
            }
        }

        return result;
    }
}
