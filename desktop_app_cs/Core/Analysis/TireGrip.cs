using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Tire grip data for one corner; port of <c>CornerTireGripData</c>.</summary>
public sealed record CornerTireGripData(
    string CornerName,
    double[] TotalG,
    double[] TireMetric,
    double MeanG,
    double MeanMetric,
    double StdG,
    double StdMetric,
    double[] BucketCenters,
    double[] BucketValues,
    long[] BucketCounts,
    double Percentile);

/// <summary>All four corners; port of <c>TireGripResult</c>.</summary>
public sealed record TireGripResult(
    CornerTireGripData FrontLeft,
    CornerTireGripData FrontRight,
    CornerTireGripData RearLeft,
    CornerTireGripData RearRight,
    string MetricMode,
    string MetricUnit,
    string AccelUnit)
{
    /// <summary>FL, FR, RL, RR.</summary>
    public IEnumerable<CornerTireGripData> Corners => new[] { FrontLeft, FrontRight, RearLeft, RearRight };
}

/// <summary>Total G against tire pressure or temperature per corner; port of <c>tire_grip.py</c>.</summary>
public static class TireGrip
{
    private static readonly string[] PressureKeys = { "tpms_press_fl", "tpms_press_fr", "tpms_press_rl", "tpms_press_rr" };
    private static readonly string[] TemperatureKeys = { "tpms_temp_fl", "tpms_temp_fr", "tpms_temp_rl", "tpms_temp_rr" };
    private static readonly string[] CornerLabels = { "FL", "FR", "RL", "RR" };

    /// <summary><c>_compute_corner</c>: moments plus the percentile of total G in equal-width tire-metric buckets.</summary>
    public static CornerTireGripData ComputeCorner(double[] totalG, double[] tireMetric, string cornerName, int numBuckets = 20, double percentile = 99.9, int minCount = 5)
    {
        var n = totalG.Length;
        if (n == 0)
        {
            return new CornerTireGripData(cornerName, totalG, tireMetric, 0, 0, 0, 0, Array.Empty<double>(), Array.Empty<double>(), Array.Empty<long>(), percentile);
        }

        var meanG = Numerics.Mean(totalG);
        var meanMetric = Numerics.Mean(tireMetric);
        var stdG = Numerics.Std(totalG);
        var stdMetric = Numerics.Std(tireMetric);

        var edges = Numerics.Linspace(Numerics.Min(tireMetric), Numerics.Max(tireMetric), numBuckets + 1);
        var bins = new int[n];
        for (var i = 0; i < n; i++)
        {
            bins[i] = Math.Clamp(Numerics.Digitize(edges, tireMetric[i]), 1, numBuckets);
        }

        var centers = new List<double>();
        var values = new List<double>();
        var counts = new List<long>();
        for (var b = 1; b <= numBuckets; b++)
        {
            var members = new List<double>();
            for (var i = 0; i < n; i++)
            {
                if (bins[i] == b)
                {
                    members.Add(totalG[i]);
                }
            }

            if (members.Count >= minCount)
            {
                centers.Add((edges[b - 1] + edges[b]) / 2);
                values.Add(Numerics.Percentile(members.ToArray(), percentile));
                counts.Add(members.Count);
            }
        }

        return new CornerTireGripData(cornerName, totalG, tireMetric, meanG, meanMetric, stdG, stdMetric, centers.ToArray(), values.ToArray(), counts.ToArray(), percentile);
    }

    private static string[] MetricKeys(string metricMode) => metricMode switch
    {
        "pressure" => PressureKeys,
        "temperature" => TemperatureKeys,
        _ => throw new ArgumentException($"metric_mode must be 'pressure' or 'temperature', got '{metricMode}'", nameof(metricMode)),
    };

    /// <summary>One lap's total G and per-corner tire metric, aligned on the lateral-G channel.</summary>
    private static (double[] TotalG, double[][] Metrics, string MetricUnit, string AccelUnit) Collect(LogFile lap, IReadOnlyDictionary<string, string> channelNames, string[] metricKeys)
    {
        var latCh = channelNames["lateral_g"];
        var inlineCh = channelNames["inline_g"];
        var metricChannels = metricKeys.Select(k => channelNames[k]).ToArray();
        var aligned = lap.SelectChannels(new[] { latCh, inlineCh }.Concat(metricChannels)).ResampleToChannel(latCh);
        var lateral = aligned[latCh].Values;
        var inline = aligned[inlineCh].Values;
        var totalG = new double[lateral.Length];
        for (var i = 0; i < totalG.Length; i++)
        {
            totalG[i] = Math.Sqrt(lateral[i] * lateral[i] + inline[i] * inline[i]);
        }

        var metrics = metricChannels.Select(ch => aligned[ch].Values).ToArray();
        return (totalG, metrics, aligned[metricChannels[0]].Units, aligned[latCh].Units);
    }

    /// <summary><c>analyze_tire_grip</c> on a lap-filtered session.</summary>
    public static TireGripResult AnalyzeTireGrip(LogFile lap, IReadOnlyDictionary<string, string> channelNames, string metricMode = "pressure", int numBuckets = 20, double percentile = 99.9)
    {
        var keys = MetricKeys(metricMode);
        var (totalG, metrics, metricUnit, accelUnit) = Collect(lap, channelNames, keys);
        var corners = CornerLabels.Select((label, c) => ComputeCorner(totalG, metrics[c], label, numBuckets, percentile)).ToArray();
        return new TireGripResult(corners[0], corners[1], corners[2], corners[3], metricMode, metricUnit, accelUnit);
    }

    /// <summary><c>analyze_tire_grip_multi_lap</c>: laps concatenated; laps that fail are skipped.</summary>
    public static TireGripResult AnalyzeTireGripMultiLap(LogFile log, IReadOnlyList<int> lapNumbers, IReadOnlyDictionary<string, string> channelNames, string metricMode = "pressure", int numBuckets = 20, double percentile = 99.9)
    {
        if (lapNumbers.Count == 0)
        {
            throw new ArgumentException("lap_numbers cannot be empty", nameof(lapNumbers));
        }

        var keys = MetricKeys(metricMode);
        var allTotalG = new List<double[]>();
        var allMetrics = new List<double[]>[] { new(), new(), new(), new() };
        var metricUnit = string.Empty;
        var accelUnit = string.Empty;
        foreach (var lapNum in lapNumbers)
        {
            try
            {
                var (totalG, metrics, mu, au) = Collect(log.FilterByLap(lapNum), channelNames, keys);
                if (metricUnit.Length == 0) metricUnit = mu;
                if (accelUnit.Length == 0) accelUnit = au;
                allTotalG.Add(totalG);
                for (var c = 0; c < 4; c++)
                {
                    allMetrics[c].Add(metrics[c]);
                }
            }
            catch (Exception e) when (e is KeyNotFoundException or ArgumentException)
            {
                continue;
            }
        }

        var combinedG = allTotalG.SelectMany(g => g).ToArray();
        var corners = CornerLabels.Select((label, c) => ComputeCorner(combinedG, allMetrics[c].SelectMany(m => m).ToArray(), label, numBuckets, percentile)).ToArray();
        return new TireGripResult(corners[0], corners[1], corners[2], corners[3], metricMode, metricUnit, accelUnit);
    }
}
