using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>A distance range on the track, metres.</summary>
public readonly record struct Zone(double Start, double End);

/// <summary>A braking, corner or acceleration segment; port of <c>TrackSegment</c>.</summary>
public sealed record TrackSegment(
    int Id,
    string SegmentType,
    double StartDist,
    double EndDist,
    string Name,
    int CornerId,
    double? ApexDist = null)
{
    /// <summary>Segment length in metres.</summary>
    public double Length => EndDist - StartDist;
}

/// <summary>One row of <c>compute_segment_stats</c>; metrics absent for a segment type are null.</summary>
public sealed record SegmentStat(
    int SegmentId,
    string SegmentName,
    string SegmentType,
    int CornerId,
    int LapNum,
    double LapTime)
{
    public double? BrakingPoint { get; init; }
    public double? BrakeOffset { get; init; }
    public double? PeakBrake { get; init; }
    public double? PeakBrakeDist { get; init; }
    public double? BrakeReleasePoint { get; init; }
    public double? BrakeReleaseOffset { get; init; }
    public double? EntrySpeed { get; init; }
    public double? BrakingDistance { get; init; }
    public double? MeanDecelG { get; init; }
    public double? MinSpeed { get; init; }
    public double? ExitSpeed { get; init; }
    public double? ThrottlePoint { get; init; }
    public double? ThrottleOffset { get; init; }
}

/// <summary>One row of <c>compute_g_utilization</c>.</summary>
public sealed record GUtilization(
    int CornerId,
    string CornerName,
    int LapNum,
    double TotalGMean,
    double TotalGMax,
    double TotalGMin,
    double TotalGMinDist,
    string TotalGMinPhase,
    double GUtilizationPct,
    double? EarlyBrakingCoastM,
    double? BrakingGMean,
    double? EntryGMean,
    double? MidGMean,
    double? ExitGMean);

/// <summary>
/// Braking/acceleration zone detection, track segments and per-segment
/// statistics; port of <c>zones.py</c>. Array inputs are per-lap channels
/// resampled onto the lap's <c>distance_m</c> timecodes.
/// </summary>
public static class Zones
{
    /// <summary>
    /// <c>identify_zones_single_lap</c>: runs of brake pressure above the
    /// threshold, and of throttle above its threshold with brake below,
    /// with accel runs bridged across gear-change gaps shorter than
    /// <paramref name="gearChangeTime"/> seconds (gap distance / mean speed).
    /// </summary>
    public static (List<Zone> Braking, List<Zone> Accel) IdentifyZonesSingleLap(
        ReadOnlySpan<double> distance,
        ReadOnlySpan<double> brake,
        ReadOnlySpan<double> throttle,
        ReadOnlySpan<double> speed,
        double? brakeThreshold = null,
        double? throttleThreshold = null,
        double gearChangeTime = 1.5)
    {
        var brakeT = brakeThreshold ?? 0.05 * Numerics.InferChannelScale(brake);
        var throttleT = throttleThreshold ?? 0.20 * Numerics.InferChannelScale(throttle);
        var n = distance.Length;

        var braking = new List<Zone>();
        int? brakeStart = null;
        for (var i = 0; i < n; i++)
        {
            var isBraking = brake[i] > brakeT;
            if (isBraking && brakeStart is null)
            {
                brakeStart = i;
            }
            else if (!isBraking && brakeStart is not null)
            {
                braking.Add(new Zone(distance[brakeStart.Value], distance[i - 1]));
                brakeStart = null;
            }
        }

        if (brakeStart is not null)
        {
            braking.Add(new Zone(distance[brakeStart.Value], distance[n - 1]));
        }

        var accelWithIdx = new List<(double Start, double End, int EndIdx)>();
        int? accelStart = null;
        for (var i = 0; i < n; i++)
        {
            var isAccel = throttle[i] > throttleT && brake[i] < brakeT;
            if (isAccel && accelStart is null)
            {
                accelStart = i;
            }
            else if (!isAccel && accelStart is not null)
            {
                accelWithIdx.Add((distance[accelStart.Value], distance[i - 1], i - 1));
                accelStart = null;
            }
        }

        if (accelStart is not null)
        {
            accelWithIdx.Add((distance[accelStart.Value], distance[n - 1], n - 1));
        }

        if (accelWithIdx.Count <= 1)
        {
            return (braking, accelWithIdx.Select(z => new Zone(z.Start, z.End)).ToList());
        }

        var merged = new List<(double Start, double End, int EndIdx)> { accelWithIdx[0] };
        foreach (var (start, end, endIdx) in accelWithIdx.Skip(1))
        {
            var (prevStart, prevEnd, prevEndIdx) = merged[^1];
            var gapDistance = start - prevEnd;
            var startIdx = Numerics.SearchSortedLeft(distance, start);
            var gapTime = double.PositiveInfinity;
            if (startIdx > prevEndIdx && startIdx < speed.Length)
            {
                var gapSpeed = Numerics.Mean(speed[prevEndIdx..(startIdx + 1)]);
                gapTime = gapSpeed > 0 ? gapDistance / gapSpeed : double.PositiveInfinity;
            }

            if (gapTime <= gearChangeTime && !braking.Any(bz => bz.Start < start && bz.End > prevEnd))
            {
                merged[^1] = (prevStart, end, endIdx);
                continue;
            }

            merged.Add((start, end, endIdx));
        }

        return (braking, merged.Select(z => new Zone(z.Start, z.End)).ToList());
    }

    /// <summary>
    /// <c>average_zones_across_laps</c>: grid vote, a point is in a zone when
    /// at least <paramref name="threshold"/> of the laps agree.
    /// </summary>
    public static (List<Zone> Braking, List<Zone> Accel) AverageZonesAcrossLaps(
        IReadOnlyList<IReadOnlyList<Zone>> allBraking,
        IReadOnlyList<IReadOnlyList<Zone>> allAccel,
        double trackLength,
        double resolution = 1.0,
        double threshold = 0.5)
    {
        var nPoints = (int)(trackLength / resolution) + 1;
        var brakeCounts = new double[nPoints];
        var accelCounts = new double[nPoints];

        static void Count(double[] counts, IReadOnlyList<IReadOnlyList<Zone>> zones, double resolution, int nPoints)
        {
            foreach (var lap in zones)
            {
                foreach (var zone in lap)
                {
                    var startIdx = Math.Max((int)(zone.Start / resolution), 0);
                    var endIdx = Math.Min((int)(zone.End / resolution) + 1, nPoints);
                    for (var i = startIdx; i < endIdx; i++)
                    {
                        counts[i] += 1;
                    }
                }
            }
        }

        Count(brakeCounts, allBraking, resolution, nPoints);
        Count(accelCounts, allAccel, resolution, nPoints);

        var minLaps = allBraking.Count * threshold;
        return (ExtractZones(brakeCounts, minLaps, resolution), ExtractZones(accelCounts, minLaps, resolution));
    }

    private static List<Zone> ExtractZones(double[] counts, double minCount, double resolution)
    {
        var zones = new List<Zone>();
        int? zoneStart = null;
        for (var i = 0; i < counts.Length; i++)
        {
            var isZone = counts[i] >= minCount;
            if (isZone && zoneStart is null)
            {
                zoneStart = i;
            }
            else if (!isZone && zoneStart is not null)
            {
                zones.Add(new Zone(zoneStart.Value * resolution, (i - 1) * resolution));
                zoneStart = null;
            }
        }

        if (zoneStart is not null)
        {
            zones.Add(new Zone(zoneStart.Value * resolution, (counts.Length - 1) * resolution));
        }

        return zones;
    }

    /// <summary>
    /// <c>merge_accel_zones_by_time</c>: join accel zones whose gap takes
    /// under <paramref name="maxGapTime"/> seconds at the reference lap's
    /// mean positive speed and contains no braking zone.
    /// </summary>
    public static List<Zone> MergeAccelZonesByTime(
        IReadOnlyList<Zone> accelZones,
        IReadOnlyList<Zone> brakingZones,
        ReadOnlySpan<double> distance,
        ReadOnlySpan<double> speed,
        double maxGapTime = 1.5)
    {
        if (accelZones.Count <= 1)
        {
            return accelZones.ToList();
        }

        var merged = new List<Zone> { accelZones[0] };
        foreach (var (start, end) in accelZones.Skip(1))
        {
            var (prevStart, prevEnd) = merged[^1];
            var gapDistance = start - prevEnd;
            var prevEndIdx = Numerics.SearchSortedLeft(distance, prevEnd);
            var startIdx = Numerics.SearchSortedLeft(distance, start);

            var gapTime = double.PositiveInfinity;
            if (startIdx > prevEndIdx && startIdx < speed.Length)
            {
                var gapSpeeds = speed[prevEndIdx..(startIdx + 1)];
                var positive = new List<double>();
                foreach (var v in gapSpeeds)
                {
                    if (v > 0)
                    {
                        positive.Add(v);
                    }
                }

                var avgSpeed = positive.Count > 0 ? Numerics.Mean(positive.ToArray()) : 0;
                gapTime = avgSpeed > 0 ? gapDistance / avgSpeed : double.PositiveInfinity;
            }

            var gapHasBraking = brakingZones.Any(bz => bz.Start < start && bz.End > prevEnd);
            if (gapTime <= maxGapTime && !gapHasBraking)
            {
                merged[^1] = new Zone(prevStart, end);
            }
            else
            {
                merged.Add(new Zone(start, end));
            }
        }

        return merged;
    }

    /// <summary>
    /// <c>create_track_segments</c>: braking, corner and acceleration segments
    /// per corner, sorted by start distance.
    /// </summary>
    public static List<TrackSegment> CreateTrackSegments(
        IReadOnlyList<Corner> corners,
        IReadOnlyList<Zone> brakingZones,
        IReadOnlyList<Zone> accelZones,
        double trackLength)
    {
        var segments = new List<TrackSegment>();
        var segmentId = 0;

        foreach (var corner in corners.OrderBy(c => c.StartDist))
        {
            double? brakeStart = null;
            foreach (var bz in brakingZones)
            {
                if (bz.End >= corner.StartDist - 100 && bz.Start < corner.StartDist)
                {
                    brakeStart = bz.Start;
                    break;
                }
            }

            brakeStart ??= Math.Max(0, corner.StartDist - 100);

            segments.Add(new TrackSegment(++segmentId, "braking", brakeStart.Value, corner.StartDist, $"{corner.Name} Braking", corner.Id));
            segments.Add(new TrackSegment(++segmentId, "corner", corner.StartDist, corner.EndDist, corner.Name, corner.Id, corner.ApexDist));

            var accelEnd = corner.EndDist + 50;
            foreach (var az in accelZones)
            {
                if (az.Start <= corner.EndDist + 50 && az.End > corner.EndDist)
                {
                    accelEnd = az.End;
                    break;
                }
            }

            foreach (var bz in brakingZones)
            {
                if (bz.Start > corner.EndDist && bz.Start < accelEnd)
                {
                    accelEnd = bz.Start;
                    break;
                }
            }

            segments.Add(new TrackSegment(++segmentId, "acceleration", corner.EndDist, Math.Max(corner.EndDist + 10, accelEnd), $"{corner.Name} Exit", corner.Id));
        }

        return segments.OrderBy(s => s.StartDist).ToList();
    }

    /// <summary>Per-lap zone-detection inputs resampled onto the lap's distance channel.</summary>
    public sealed record LapArrays(int LapNum, double LapTimeS, double[] Distance, double[] Brake, double[] Throttle, double[] SpeedMs, double[] SpeedKmh);

    /// <summary>
    /// The per-lap arrays <c>detect_zones_averaged</c> / <c>compute_segment_stats</c>
    /// extract: filter by lap, select the channels, resample onto <c>distance_m</c>.
    /// </summary>
    public static LapArrays ExtractLap(LogFile log, int lapNum, string throttleChannel, string brakeChannel, string speedChannel)
    {
        var lap = log.Laps.First(l => l.Num == lapNum);
        var aligned = log.FilterByLap(lapNum)
            .SelectChannels(new[] { "distance_m", brakeChannel, throttleChannel, speedChannel, "speed_kmh" }.Distinct())
            .ResampleToChannel("distance_m");
        return new LapArrays(
            lapNum,
            lap.LapTimeS,
            aligned["distance_m"].Values,
            aligned[brakeChannel].Values,
            aligned[throttleChannel].Values,
            aligned[speedChannel].Values,
            aligned["speed_kmh"].Values);
    }

    /// <summary><c>detect_zones_from_arrays</c>: per-lap zones, averaged, accel merged on the reference (first) lap.</summary>
    public static (List<Zone> Braking, List<Zone> Accel) DetectZonesFromArrays(
        IReadOnlyList<LapArrays> laps,
        double resolution = 1.0,
        double threshold = 0.5,
        double maxGapTime = 1.5,
        double? brakeThreshold = null,
        double? throttleThreshold = null)
    {
        var allBraking = new List<IReadOnlyList<Zone>>();
        var allAccel = new List<IReadOnlyList<Zone>>();
        LapArrays? reference = null;

        foreach (var lap in laps)
        {
            if (lap.Distance.Length < 10)
            {
                continue;
            }

            reference ??= lap;
            var (braking, accel) = IdentifyZonesSingleLap(lap.Distance, lap.Brake, lap.Throttle, lap.SpeedMs, brakeThreshold, throttleThreshold);
            allBraking.Add(braking);
            allAccel.Add(accel);
        }

        if (allBraking.Count == 0 || reference is null)
        {
            return (new List<Zone>(), new List<Zone>());
        }

        var trackLength = Numerics.Max(reference.Distance);
        var (brakingZones, accelZones) = AverageZonesAcrossLaps(allBraking, allAccel, trackLength, resolution, threshold);
        accelZones = MergeAccelZonesByTime(accelZones, brakingZones, reference.Distance, reference.SpeedMs, maxGapTime);
        return (brakingZones, accelZones);
    }

    /// <summary><c>detect_zones_averaged</c> over the given laps of a session.</summary>
    public static (List<Zone> Braking, List<Zone> Accel) DetectZonesAveraged(
        LogFile log,
        IEnumerable<int> lapNums,
        string throttleChannel,
        string brakeChannel,
        string speedChannel,
        double resolution = 1.0,
        double threshold = 0.5,
        double maxGapTime = 1.5,
        double? brakeThreshold = null,
        double? throttleThreshold = null)
    {
        var laps = lapNums.Select(n => ExtractLap(log, n, throttleChannel, brakeChannel, speedChannel)).ToList();
        return DetectZonesFromArrays(laps, resolution, threshold, maxGapTime, brakeThreshold, throttleThreshold);
    }

    /// <summary><c>compute_segment_stats_from_arrays</c>; speeds are km/h.</summary>
    public static List<SegmentStat> ComputeSegmentStats(
        IReadOnlyList<LapArrays> laps,
        IReadOnlyList<TrackSegment> segments,
        double? brakeThreshold = null,
        double? throttleThreshold = null)
    {
        var stats = new List<SegmentStat>();
        var resolved = false;
        var brakeT = brakeThreshold ?? 0;
        var throttleT = throttleThreshold ?? 0;

        foreach (var lap in laps)
        {
            if (lap.Distance.Length < 10)
            {
                continue;
            }

            if (!resolved)
            {
                brakeT = brakeThreshold ?? 0.05 * Numerics.InferChannelScale(lap.Brake);
                throttleT = throttleThreshold ?? 0.20 * Numerics.InferChannelScale(lap.Throttle);
                resolved = true;
            }

            ReadOnlySpan<double> distance = lap.Distance;
            ReadOnlySpan<double> speed = lap.SpeedKmh;
            ReadOnlySpan<double> brake = lap.Brake;
            ReadOnlySpan<double> throttle = lap.Throttle;

            for (var segIdx = 0; segIdx < segments.Count; segIdx++)
            {
                var seg = segments[segIdx];
                var stat = new SegmentStat(seg.Id, seg.Name, seg.SegmentType, seg.CornerId, lap.LapNum, lap.LapTimeS);

                switch (seg.SegmentType)
                {
                    case "braking":
                        {
                            var brakingPoint = FindBrakingPoint(distance, brake, seg.StartDist, seg.EndDist, brakeT);
                            stat = stat with { BrakingPoint = brakingPoint, BrakeOffset = brakingPoint - seg.StartDist };

                            var peak = FindPeakBrake(brake, distance, seg.StartDist, seg.EndDist);
                            if (peak is { } p)
                            {
                                stat = stat with { PeakBrake = p.Value, PeakBrakeDist = p.Dist };
                            }

                            var releaseEnd = seg.EndDist;
                            if (segIdx + 1 < segments.Count && segments[segIdx + 1].SegmentType == "corner")
                            {
                                releaseEnd = segments[segIdx + 1].EndDist;
                            }

                            var release = FindBrakeRelease(brake, distance, seg.StartDist, releaseEnd);
                            stat = stat with { BrakeReleasePoint = release, BrakeReleaseOffset = release - seg.StartDist };

                            if (brakingPoint is { } bp)
                            {
                                var bpIdx = Numerics.SearchSortedLeft(distance, bp);
                                if (bpIdx < speed.Length)
                                {
                                    stat = stat with { EntrySpeed = speed[bpIdx] };
                                }
                            }

                            if (brakingPoint is { } bp2 && release is { } rel)
                            {
                                stat = stat with
                                {
                                    BrakingDistance = rel - bp2,
                                    MeanDecelG = ComputeDeceleration(speed, distance, bp2, rel),
                                };
                            }

                            break;
                        }

                    case "corner":
                        stat = stat with
                        {
                            MinSpeed = FindMinSpeed(distance, speed, seg.StartDist, seg.EndDist),
                            ExitSpeed = FindExitSpeed(distance, speed, seg.StartDist, seg.EndDist),
                        };
                        break;

                    case "acceleration":
                        {
                            var throttlePoint = FindThrottlePoint(distance, throttle, brake, seg.StartDist, seg.EndDist, throttleT, brakeT);
                            stat = stat with { ThrottlePoint = throttlePoint, ThrottleOffset = throttlePoint - seg.StartDist };
                            break;
                        }
                }

                stats.Add(stat);
            }
        }

        return stats;
    }

    /// <summary>Segment [start, end] as index range [si, ei) on a sorted distance array (searchsorted left / right).</summary>
    private static (int Si, int Ei) Range(ReadOnlySpan<double> distance, double startDist, double endDist) =>
        (Numerics.SearchSortedLeft(distance, startDist), Numerics.SearchSortedRight(distance, endDist));

    /// <summary><c>_find_braking_point_np</c>.</summary>
    public static double? FindBrakingPoint(ReadOnlySpan<double> distance, ReadOnlySpan<double> brake, double startDist, double endDist, double brakeThreshold)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        for (var i = si; i < ei; i++)
        {
            if (brake[i] > brakeThreshold)
            {
                return distance[i];
            }
        }

        return null;
    }

    /// <summary><c>_find_peak_brake_np</c>.</summary>
    public static (double Value, double Dist)? FindPeakBrake(ReadOnlySpan<double> brake, ReadOnlySpan<double> distance, double startDist, double endDist)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        if (si >= ei)
        {
            return null;
        }

        var peak = si + Numerics.ArgMax(brake[si..ei]);
        return (brake[peak], distance[peak]);
    }

    /// <summary><c>_find_brake_release_np</c>: first sample after the peak below <paramref name="thresholdPct"/> of it.</summary>
    public static double? FindBrakeRelease(ReadOnlySpan<double> brake, ReadOnlySpan<double> distance, double startDist, double endDist, double thresholdPct = 0.10)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        if (si >= ei)
        {
            return null;
        }

        var peak = si + Numerics.ArgMax(brake[si..ei]);
        var peakValue = brake[peak];
        if (peakValue <= 0)
        {
            return null;
        }

        var releaseThreshold = thresholdPct * peakValue;
        for (var i = peak; i < ei; i++)
        {
            if (brake[i] < releaseThreshold)
            {
                return distance[i];
            }
        }

        return null;
    }

    /// <summary><c>_compute_deceleration</c>: mean deceleration in g from v² = v0² − 2ad; speeds in km/h.</summary>
    public static double? ComputeDeceleration(ReadOnlySpan<double> speedKmh, ReadOnlySpan<double> distance, double startDist, double endDist)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        if (si >= ei || ei > speedKmh.Length)
        {
            return null;
        }

        var v0 = speedKmh[si] / 3.6;
        var v1 = speedKmh[ei - 1] / 3.6;
        var d = distance[ei - 1] - distance[si];
        if (d <= 0)
        {
            return null;
        }

        var decel = (v0 * v0 - v1 * v1) / (2 * d) / 9.81;
        return decel > 0 ? decel : null;
    }

    /// <summary><c>_find_throttle_point_np</c>.</summary>
    public static double? FindThrottlePoint(ReadOnlySpan<double> distance, ReadOnlySpan<double> throttle, ReadOnlySpan<double> brake, double startDist, double endDist, double throttleThreshold, double brakeThreshold)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        for (var i = si; i < ei; i++)
        {
            if (throttle[i] > throttleThreshold && brake[i] < brakeThreshold)
            {
                return distance[i];
            }
        }

        return null;
    }

    /// <summary><c>_find_min_speed_np</c>.</summary>
    public static double? FindMinSpeed(ReadOnlySpan<double> distance, ReadOnlySpan<double> speed, double startDist, double endDist)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        return si < ei ? Numerics.Min(speed[si..ei]) : null;
    }

    /// <summary><c>_find_exit_speed_np</c>.</summary>
    public static double? FindExitSpeed(ReadOnlySpan<double> distance, ReadOnlySpan<double> speed, double startDist, double endDist)
    {
        var (si, ei) = Range(distance, startDist, endDist);
        return si < ei ? speed[ei - 1] : null;
    }

    /// <summary>Per-lap G-utilization inputs: distance, speed (km/h), lateral G and optional inline G.</summary>
    public sealed record LapGArrays(int LapNum, double[] Distance, double[] SpeedKmh, double[] LateralG, double[]? InlineG);

    /// <summary>
    /// <c>compute_g_utilization</c>: total G = √(lat² + inline²) through each
    /// corner complex (braking start → accel end); the "G hole" between the
    /// braking and cornering peaks, its phase, and phase means. Inline G is
    /// derived from speed when a lap has none.
    /// </summary>
    public static List<GUtilization> ComputeGUtilization(
        IReadOnlyList<LapGArrays> laps,
        IReadOnlyList<TrackSegment> segments,
        IReadOnlyList<Corner> corners)
    {
        var brakingSegs = new Dictionary<int, TrackSegment>();
        var accelSegs = new Dictionary<int, TrackSegment>();
        foreach (var seg in segments)
        {
            if (seg.SegmentType == "braking")
            {
                brakingSegs[seg.CornerId] = seg;
            }
            else if (seg.SegmentType == "acceleration")
            {
                accelSegs[seg.CornerId] = seg;
            }
        }

        var rows = new List<GUtilization>();
        foreach (var lap in laps)
        {
            ReadOnlySpan<double> dist = lap.Distance;
            if (dist.Length < 10)
            {
                continue;
            }

            var inlineG = lap.InlineG ?? InlineGFromSpeed(lap.Distance, lap.SpeedKmh);
            var totalG = new double[dist.Length];
            for (var i = 0; i < totalG.Length; i++)
            {
                totalG[i] = Math.Sqrt(lap.LateralG[i] * lap.LateralG[i] + inlineG[i] * inlineG[i]);
            }

            foreach (var corner in corners)
            {
                var complexStart = brakingSegs.TryGetValue(corner.Id, out var brakeSeg) ? brakeSeg.StartDist : corner.StartDist;
                var complexEnd = accelSegs.TryGetValue(corner.Id, out var accelSeg) ? accelSeg.EndDist : corner.EndDist;

                var ci = Numerics.SearchSortedLeft(dist, complexStart);
                var ce = Numerics.SearchSortedRight(dist, complexEnd);
                if (ci >= ce)
                {
                    continue;
                }

                ReadOnlySpan<double> complexTotalG = totalG.AsSpan(ci, ce - ci);
                var totalGMean = Numerics.Mean(complexTotalG);
                var totalGMax = Numerics.Max(complexTotalG);

                var cornerSi = Numerics.SearchSortedLeft(dist, corner.StartDist);
                var cornerEi = Numerics.SearchSortedRight(dist, corner.EndDist);

                var brakePeakIdx = ci;
                if (cornerSi > ci)
                {
                    brakePeakIdx = ci + Numerics.ArgMax(totalG.AsSpan(ci, cornerSi - ci));
                }

                var cornerPeakIdx = cornerSi;
                if (cornerEi > cornerSi)
                {
                    cornerPeakIdx = cornerSi + Numerics.ArgMax(totalG.AsSpan(cornerSi, cornerEi - cornerSi));
                }

                var peakLo = Math.Min(brakePeakIdx, cornerPeakIdx);
                var peakHi = Math.Max(brakePeakIdx, cornerPeakIdx);
                int valleyMinIdx;
                if (peakHi > peakLo + 1)
                {
                    valleyMinIdx = peakLo + Numerics.ArgMin(totalG.AsSpan(peakLo, peakHi - peakLo));
                }
                else
                {
                    valleyMinIdx = totalG[peakLo] < totalG[peakHi] ? peakLo : peakHi;
                }

                var totalGMin = totalG[valleyMinIdx];
                var totalGMinDist = dist[valleyMinIdx];

                var cornerLen = corner.EndDist - corner.StartDist;
                var midHalf = 0.10 * cornerLen;
                var apex = corner.ApexDist;
                var phases = new (string Name, double Start, double End)[]
                {
                    ("braking", complexStart, corner.StartDist),
                    ("entry", corner.StartDist, apex),
                    ("mid", apex - midHalf, apex + midHalf),
                    ("exit", apex + midHalf, corner.EndDist),
                };

                var minPhase = "entry";
                foreach (var (name, ps, pe) in phases)
                {
                    if (ps <= totalGMinDist && totalGMinDist <= pe)
                    {
                        minPhase = name;
                        break;
                    }
                }

                var phaseMeans = new Dictionary<string, double?>();
                foreach (var (name, ps, pe) in phases)
                {
                    var pi = Numerics.SearchSortedLeft(dist, ps);
                    var pj = Numerics.SearchSortedRight(dist, pe);
                    phaseMeans[name] = pi < pj ? Numerics.Mean(totalG.AsSpan(pi, pj - pi)) : null;
                }

                var peakG = Math.Min(totalG[brakePeakIdx], totalG[cornerPeakIdx]);
                var gUtilPct = peakG > 0 ? totalGMin / Math.Max(peakG, 1e-6) * 100 : 0.0;

                double? earlyBrakingCoast = null;
                if (minPhase == "braking" && cornerSi > brakePeakIdx)
                {
                    var brakePeakG = totalG[brakePeakIdx];
                    var thresholdG = brakePeakG * 0.5;
                    for (var i = brakePeakIdx; i < cornerSi; i++)
                    {
                        if (totalG[i] < thresholdG)
                        {
                            var coast = corner.StartDist - dist[i];
                            if (coast > 0)
                            {
                                earlyBrakingCoast = Math.Round(coast, 1, MidpointRounding.ToEven);
                            }

                            break;
                        }
                    }
                }

                rows.Add(new GUtilization(
                    corner.Id,
                    corner.Name,
                    lap.LapNum,
                    totalGMean,
                    totalGMax,
                    totalGMin,
                    totalGMinDist,
                    minPhase,
                    gUtilPct,
                    earlyBrakingCoast,
                    phaseMeans["braking"],
                    phaseMeans["entry"],
                    phaseMeans["mid"],
                    phaseMeans["exit"]));
            }
        }

        return rows;
    }

    /// <summary>Longitudinal G estimated from speed over distance, smoothed with a 5-sample box (numpy order of operations).</summary>
    public static double[] InlineGFromSpeed(ReadOnlySpan<double> distance, ReadOnlySpan<double> speedKmh)
    {
        var n = distance.Length;
        var speedMs = new double[n];
        for (var i = 0; i < n; i++)
        {
            speedMs[i] = speedKmh[i] / 3.6;
        }

        var accel = new double[n];
        for (var i = 1; i < n; i++)
        {
            var dd = distance[i] - distance[i - 1];
            var avgSpd = (speedMs[i - 1] + speedMs[i]) / 2;
            var safeAvg = avgSpd > 0.1 ? avgSpd : 0.1;
            var dt = dd / safeAvg;
            var dv = speedMs[i] - speedMs[i - 1];
            var safeDt = dt > 1e-6 ? dt : 1e-6;
            accel[i] = dv / safeDt / 9.81;
        }

        if (n > 1)
        {
            accel[0] = accel[1];
        }

        var kernel = new double[] { 0.2, 0.2, 0.2, 0.2, 0.2 };
        return Numerics.ConvolveSame(accel, kernel);
    }
}
