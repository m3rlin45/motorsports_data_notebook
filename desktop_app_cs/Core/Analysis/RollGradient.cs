using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Roll-vs-lateral-G data and fit for one axle; port of <c>AxleRollData</c>.</summary>
public sealed record AxleRollData(
    string Axle,
    double[] LateralG,
    double[] RollDeg,
    double GradientDegPerG,
    double InterceptDeg,
    double R2,
    double[] BinCentersG,
    double[] BinMeanDeg,
    long[] BinCounts,
    int NSamples)
{
    /// <summary>Magnitude of the roll gradient, deg/g.</summary>
    public double AbsGradientDegPerG => Math.Abs(GradientDegPerG);
}

/// <summary>Roll gradient analysis for both axles; port of <c>RollGradientResult</c>.</summary>
public sealed record RollGradientResult(
    AxleRollData Front,
    AxleRollData Rear,
    TrackWidths TrackWidths,
    string LateralGUnit,
    double RollRatio,
    double RollRatioInterceptDeg,
    double RollRatioR2,
    double[] RatioBinCentersDeg,
    double[] RatioBinMeanDeg,
    long[] RatioBinCounts,
    double[]? XM,
    double[]? YM)
{
    /// <summary>True when every sample has a local track position.</summary>
    public bool HasTrackMap => XM is { Length: > 0 } && YM is not null;

    /// <summary>Front gradient / rear gradient (alternative roll ratio estimate).</summary>
    public double GradientRatio =>
        Rear.GradientDegPerG == 0 || !double.IsFinite(Rear.GradientDegPerG) ? double.NaN : Front.GradientDegPerG / Rear.GradientDegPerG;

    /// <summary>Mean of the front and rear gradient magnitudes.</summary>
    public double MeanGradientDegPerG => 0.5 * (Front.AbsGradientDegPerG + Rear.AbsGradientDegPerG);

    /// <summary>Front gradient magnitude minus rear.</summary>
    public double FrontMinusRearDegPerG => Front.AbsGradientDegPerG - Rear.AbsGradientDegPerG;

    /// <summary>True if front and rear roll in the same direction for a given lateral g.</summary>
    public bool SignsConsistent => Numerics.Sign(Front.GradientDegPerG) == Numerics.Sign(Rear.GradientDegPerG);
}

/// <summary>Roll gradient and roll ratio from shock pot travel; port of <c>roll_gradient.py</c>.</summary>
public static class RollGradient
{
    private const double GMs2 = 9.80665;

    /// <summary>Canonical channel keys the analysis needs.</summary>
    public static readonly string[] RequiredKeys = { "shock_fl", "shock_fr", "shock_rl", "shock_rr", "lateral_g" };

    /// <summary><c>compute_roll_angle_deg</c>: degrees(atan((left − right) / track)).</summary>
    public static double[] ComputeRollAngleDeg(ReadOnlySpan<double> leftTravelMm, ReadOnlySpan<double> rightTravelMm, double trackMm)
    {
        if (trackMm <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(trackMm), "track_mm must be positive");
        }

        var roll = new double[leftTravelMm.Length];
        for (var i = 0; i < roll.Length; i++)
        {
            roll[i] = Math.Atan((leftTravelMm[i] - rightTravelMm[i]) / trackMm) * (180.0 / Math.PI);
        }

        return roll;
    }

    /// <summary>
    /// <c>fit_roll_gradient</c>: least-squares <c>roll = gradient·g + intercept</c>
    /// over finite samples with |g| ≥ <paramref name="minLateralG"/>; all NaN
    /// with fewer than two samples or no spread.
    /// </summary>
    public static (double Gradient, double Intercept, double R2) FitRollGradient(ReadOnlySpan<double> lateralG, ReadOnlySpan<double> rollDeg, double minLateralG = 0.0)
    {
        var xs = new List<double>(lateralG.Length);
        var ys = new List<double>(lateralG.Length);
        for (var i = 0; i < lateralG.Length; i++)
        {
            if (double.IsFinite(lateralG[i]) && double.IsFinite(rollDeg[i]) && Math.Abs(lateralG[i]) >= minLateralG)
            {
                xs.Add(lateralG[i]);
                ys.Add(rollDeg[i]);
            }
        }

        if (xs.Count < 2 || xs.Max() - xs.Min() == 0)
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        var x = xs.ToArray();
        var y = ys.ToArray();
        var (slope, intercept) = Numerics.FitLine(x, y);
        var yMean = Numerics.Mean(y);
        var residSq = new double[x.Length];
        var totSq = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            var resid = y[i] - (slope * x[i] + intercept);
            residSq[i] = resid * resid;
            var d = y[i] - yMean;
            totSq[i] = d * d;
        }

        var ssTot = Numerics.Sum(totSq);
        var r2 = ssTot > 0 ? 1.0 - Numerics.Sum(residSq) / ssTot : double.NaN;
        return (slope, intercept, r2);
    }

    /// <summary>
    /// <c>bin_roll_by_lateral_g</c>: mean roll in fixed-width lateral-g bins,
    /// dropping bins with fewer than <paramref name="minCount"/> samples.
    /// </summary>
    public static (double[] Centers, double[] Means, long[] Counts) BinRollByLateralG(ReadOnlySpan<double> lateralG, ReadOnlySpan<double> rollDeg, double binSizeG = 0.1, int minCount = 5)
    {
        var xs = new List<double>();
        var ys = new List<double>();
        for (var i = 0; i < lateralG.Length; i++)
        {
            if (double.IsFinite(lateralG[i]) && double.IsFinite(rollDeg[i]))
            {
                xs.Add(lateralG[i]);
                ys.Add(rollDeg[i]);
            }
        }

        if (xs.Count == 0 || binSizeG <= 0)
        {
            return (Array.Empty<double>(), Array.Empty<double>(), Array.Empty<long>());
        }

        var lo = Math.Floor(xs.Min() / binSizeG) * binSizeG;
        var hi = Math.Ceiling(xs.Max() / binSizeG) * binSizeG + binSizeG * 0.5;
        var edges = Numerics.ARange(lo, hi, binSizeG);
        if (edges.Length < 2)
        {
            edges = new[] { lo, lo + binSizeG };
        }

        var nBins = edges.Length - 1;
        var counts = new long[nBins];
        var sums = new double[nBins];
        for (var i = 0; i < xs.Count; i++)
        {
            var idx = Math.Clamp(Numerics.Digitize(edges, xs[i]) - 1, 0, nBins - 1);
            counts[idx]++;
            sums[idx] += ys[i];
        }

        var keepMin = Math.Max(1, minCount);
        var centers = new List<double>();
        var means = new List<double>();
        var kept = new List<long>();
        for (var b = 0; b < nBins; b++)
        {
            if (counts[b] >= keepMin)
            {
                centers.Add((edges[b] + edges[b + 1]) / 2);
                means.Add(sums[b] / counts[b]);
                kept.Add(counts[b]);
            }
        }

        return (centers.ToArray(), means.ToArray(), kept.ToArray());
    }

    /// <summary><c>_smooth</c>: centred moving average with edge padding (no-op for window ≤ 1 or short input).</summary>
    public static double[] Smooth(double[] x, int window)
    {
        if (window <= 1 || x.Length < window)
        {
            return x;
        }

        var padBefore = window / 2;
        var padAfter = window - 1 - window / 2;
        var n = x.Length;
        var padded = new double[n + padBefore + padAfter];
        for (var i = 0; i < padded.Length; i++)
        {
            padded[i] = x[Math.Clamp(i - padBefore, 0, n - 1)];
        }

        var weight = 1.0 / window;
        var result = new double[padded.Length - window + 1];
        for (var i = 0; i < result.Length; i++)
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

    /// <summary><c>_lateral_to_g</c>: m/s² channels are divided by g.</summary>
    public static double[] LateralToG(double[] values, string unit)
    {
        var u = unit.ToLowerInvariant().Replace(" ", "");
        if (!u.Contains("m/s"))
        {
            return values;
        }

        return values.Select(v => v / GMs2).ToArray();
    }

    /// <summary>Per-lap aligned inputs: smoothed lateral g, front and rear roll, and optional GPS.</summary>
    public sealed record LapInputs(double[] LateralG, double[] RollFront, double[] RollRear, string Unit, double[]? Lat, double[]? Lon);

    /// <summary><c>_collect_aligned</c> for one lap-filtered session.</summary>
    public static LapInputs CollectAligned(
        LogFile lap,
        IReadOnlyDictionary<string, string> channelNames,
        MotionRatios motionRatios,
        TrackWidths trackWidths,
        int smoothingWindow,
        (string Lat, string Lon)? gps)
    {
        var fl = channelNames["shock_fl"];
        var fr = channelNames["shock_fr"];
        var rl = channelNames["shock_rl"];
        var rr = channelNames["shock_rr"];
        var latName = channelNames["lateral_g"];
        var wanted = new List<string> { fl, fr, rl, rr, latName };
        if (gps is { } g)
        {
            wanted.Add(g.Lat);
            wanted.Add(g.Lon);
        }

        var aligned = lap.SelectChannels(wanted).ResampleToChannel(fl);
        var unit = aligned[latName].Units;
        var lateral = LateralToG(aligned[latName].Values, unit);
        double[]? gpsLat = null, gpsLon = null;
        if (gps is { } g2)
        {
            gpsLat = aligned[g2.Lat].Values;
            gpsLon = aligned[g2.Lon].Values;
        }

        double[] Wheel(string name, double ratio) => Smooth(aligned[name].Values.Select(v => v / ratio).ToArray(), smoothingWindow);

        var rollFront = ComputeRollAngleDeg(Wheel(fl, motionRatios.FrontLeft), Wheel(fr, motionRatios.FrontRight), trackWidths.FrontMm);
        var rollRear = ComputeRollAngleDeg(Wheel(rl, motionRatios.RearLeft), Wheel(rr, motionRatios.RearRight), trackWidths.RearMm);
        return new LapInputs(Smooth(lateral, smoothingWindow), rollFront, rollRear, unit, gpsLat, gpsLon);
    }

    /// <summary>GPS channel names when configured and present, else null.</summary>
    public static (string Lat, string Lon)? GpsChannels(LogFile log, IReadOnlyDictionary<string, string> channelNames)
    {
        if (channelNames.TryGetValue("gps_latitude", out var lat) && channelNames.TryGetValue("gps_longitude", out var lon)
            && lat.Length > 0 && lon.Length > 0 && log.HasChannel(lat) && log.HasChannel(lon))
        {
            return (lat, lon);
        }

        return null;
    }

    private static void CheckInputs(LogFile log, IReadOnlyDictionary<string, string> channelNames)
    {
        var missing = RequiredKeys.Where(k => !channelNames.ContainsKey(k)).ToList();
        if (missing.Count > 0)
        {
            throw new KeyNotFoundException($"Missing channel keys: {string.Join(", ", missing)}");
        }

        if (Suspension.ChannelIsVelocity(log, channelNames["shock_fl"]))
        {
            throw new InvalidOperationException("Shock channels contain velocities (e.g. iRacing); roll gradient needs shock pot displacement");
        }
    }

    /// <summary><c>analyze_roll_gradient</c> on a lap-filtered session.</summary>
    public static RollGradientResult AnalyzeRollGradient(
        LogFile lap,
        IReadOnlyDictionary<string, string> channelNames,
        MotionRatios motionRatios,
        TrackWidths trackWidths,
        int smoothingWindow = 5,
        double minLateralG = 0.0,
        double binSizeG = 0.1)
    {
        CheckInputs(lap, channelNames);
        var inputs = CollectAligned(lap, channelNames, motionRatios, trackWidths, smoothingWindow, GpsChannels(lap, channelNames));
        return BuildResult(inputs.LateralG, inputs.RollFront, inputs.RollRear, trackWidths, inputs.Unit, minLateralG, binSizeG, inputs.Lat, inputs.Lon);
    }

    /// <summary><c>analyze_roll_gradient_multi_lap</c>: samples of all laps concatenated before fitting; unprocessable laps skipped.</summary>
    public static RollGradientResult AnalyzeRollGradientMultiLap(
        LogFile log,
        IReadOnlyList<int> lapNumbers,
        IReadOnlyDictionary<string, string> channelNames,
        MotionRatios motionRatios,
        TrackWidths trackWidths,
        int smoothingWindow = 5,
        double minLateralG = 0.0,
        double binSizeG = 0.1)
    {
        if (lapNumbers.Count == 0)
        {
            throw new ArgumentException("lap_numbers cannot be empty", nameof(lapNumbers));
        }

        CheckInputs(log, channelNames);
        var gps = GpsChannels(log, channelNames);
        var parts = new List<LapInputs>();
        foreach (var lapNum in lapNumbers)
        {
            try
            {
                parts.Add(CollectAligned(log.FilterByLap(lapNum), channelNames, motionRatios, trackWidths, smoothingWindow, gps));
            }
            catch (Exception e) when (e is KeyNotFoundException or ArgumentException or InvalidOperationException)
            {
                continue;
            }
        }

        if (parts.Count == 0)
        {
            throw new InvalidOperationException("No laps could be processed for roll gradient analysis");
        }

        var unit = parts.Select(p => p.Unit).FirstOrDefault(u => u.Length > 0) ?? string.Empty;
        var haveGps = gps is not null && parts.All(p => p.Lat is not null && p.Lon is not null);
        return BuildResult(
            parts.SelectMany(p => p.LateralG).ToArray(),
            parts.SelectMany(p => p.RollFront).ToArray(),
            parts.SelectMany(p => p.RollRear).ToArray(),
            trackWidths,
            unit,
            minLateralG,
            binSizeG,
            haveGps ? parts.SelectMany(p => p.Lat!).ToArray() : null,
            haveGps ? parts.SelectMany(p => p.Lon!).ToArray() : null);
    }

    /// <summary><c>_build_result</c>: per-axle fits plus the front-vs-rear roll ratio fit and binning.</summary>
    public static RollGradientResult BuildResult(
        double[] lateral,
        double[] rollFront,
        double[] rollRear,
        TrackWidths trackWidths,
        string unit,
        double minLateralG,
        double binSizeG,
        double[]? lat = null,
        double[]? lon = null)
    {
        double[]? xM = null, yM = null;
        if (lat is not null && lon is not null && lat.Length == lateral.Length && lat.Length > 0)
        {
            (xM, yM) = Corners.GpsToLocalXy(lat, lon);
        }

        var front = BuildAxle("front", lateral, rollFront, minLateralG, binSizeG);
        var rear = BuildAxle("rear", lateral, rollRear, minLateralG, binSizeG);

        var maskedRear = new List<double>();
        var maskedFront = new List<double>();
        for (var i = 0; i < lateral.Length; i++)
        {
            if (double.IsFinite(lateral[i]) && Math.Abs(lateral[i]) >= minLateralG)
            {
                maskedRear.Add(rollRear[i]);
                maskedFront.Add(rollFront[i]);
            }
        }

        var (ratio, ratioIntercept, ratioR2) = FitRollGradient(maskedRear.ToArray(), maskedFront.ToArray());
        var rearBin = Math.Abs(rear.GradientDegPerG) * binSizeG;
        if (!double.IsFinite(rearBin) || rearBin <= 0)
        {
            rearBin = 0.1;
        }

        var (centers, means, counts) = BinRollByLateralG(maskedRear.ToArray(), maskedFront.ToArray(), rearBin);
        return new RollGradientResult(front, rear, trackWidths, unit, ratio, ratioIntercept, ratioR2, centers, means, counts, xM, yM);
    }

    private static AxleRollData BuildAxle(string axle, double[] lateralG, double[] rollDeg, double minLateralG, double binSizeG)
    {
        var (gradient, intercept, r2) = FitRollGradient(lateralG, rollDeg, minLateralG);
        var (centers, means, counts) = BinRollByLateralG(lateralG, rollDeg, binSizeG);
        var used = lateralG.Count(g => double.IsFinite(g) && Math.Abs(g) >= minLateralG);
        return new AxleRollData(axle, lateralG, rollDeg, gradient, intercept, r2, centers, means, counts, used);
    }
}
