using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>Per-lap trace slice around a corner, for the detail overlay.</summary>
public sealed record LapTrace(int LapNum, double[] Distance, double[] Throttle, double[] Brake, double[] LateralG);

/// <summary>Consistency of one corner across laps; port of the desktop app's <c>CornerConsistencyData</c>.</summary>
public sealed class CornerConsistency
{
    /// <summary>The corner.</summary>
    public required Corner Corner { get; init; }

    /// <summary>Throttle acceptance % per lap where it was found.</summary>
    public List<double> TaValues { get; } = new();
    public double TaMean { get; set; }
    public double TaStd { get; set; }

    /// <summary>Braking point per lap (distance along the track, metres).</summary>
    public List<double> BpValues { get; } = new();
    public double BpStd { get; set; }

    /// <summary>Minimum corner speed per lap, km/h.</summary>
    public List<double> SpeedValues { get; } = new();
    public double SpeedMean { get; set; }
    public double SpeedStd { get; set; }

    /// <summary>Corner exit speed per lap, km/h.</summary>
    public List<double> ExitSpeedValues { get; } = new();
    public double ExitSpeedMean { get; set; }
    public double ExitSpeedStd { get; set; }

    /// <summary>Length of the acceleration segment after the corner, metres.</summary>
    public double AccelZoneLength { get; set; }

    /// <summary>Exit-speed spread × acceleration zone length: where consistency pays most.</summary>
    public double OpportunityScore { get; set; }

    /// <summary>Per-lap traces from before the braking zone to past the corner exit.</summary>
    public List<LapTrace> LapTraces { get; } = new();

    /// <summary>Start of the braking segment, when one exists.</summary>
    public double? BrakingStart { get; set; }
}

/// <summary>Everything the Driver Consistency tab shows; port of <c>DriverConsistencyResult</c>.</summary>
public sealed record DriverConsistencyResult(
    List<Corner> Corners,
    List<CornerConsistency> CornerData,
    List<TrackSegment> Segments,
    List<SegmentStat> Stats,
    double[] RefLat,
    double[] RefLon,
    double[] RefDistance);

/// <summary>
/// The desktop app's driver-consistency pipeline (corners from the fastest
/// selected lap, zones and segment stats across the selected laps, throttle
/// acceptance per lap and corner, per-corner aggregates and traces); port of
/// <c>desktop_app/.../driver/analysis/driver_consistency.py</c>.
/// </summary>
public static class DriverConsistency
{
    /// <summary>Channel keys the pipeline reads from the profile mapping.</summary>
    public static readonly string[] RequiredKeys = { "throttle", "brake", "lateral_g", "gps_latitude", "gps_longitude", "gps_speed" };

    /// <summary>Run the pipeline over <paramref name="selectedLaps"/>; <paramref name="cancellation"/> is checked between laps.</summary>
    /// <exception cref="InvalidOperationException">No selected lap exists, or no corner was detected.</exception>
    public static DriverConsistencyResult Analyze(
        LogFile log,
        IReadOnlyList<int> selectedLaps,
        IReadOnlyDictionary<string, string> channelNames,
        double cornerThreshold = Corners.DefaultThreshold,
        double throttleThreshold = 98.0,
        double sustainTimeMs = 500.0,
        CancellationToken cancellation = default)
    {
        var selected = log.Laps.Where(l => selectedLaps.Contains(l.Num)).ToList();
        if (selected.Count == 0)
        {
            throw new InvalidOperationException("No valid laps selected");
        }

        var latName = channelNames["gps_latitude"];
        var lonName = channelNames["gps_longitude"];

        // Step 1: the fastest selected lap's GPS is the corner reference.
        var bestLap = selected.MinBy(l => l.LapTimeMs)!;
        var best = log.FilterByLap(bestLap.Num).SelectChannels(new[] { latName, lonName, "distance_m" }).ResampleToChannel(latName);
        var lat = best[latName].Values;
        var lon = best[lonName].Values;
        var refDist = best["distance_m"].Values;
        var valid = Enumerable.Range(0, lat.Length).Where(i => lat[i] != 0.0 || lon[i] != 0.0).ToArray();
        if (valid.Length != lat.Length)
        {
            refDist = valid.Select(i => refDist[i]).ToArray();
        }

        // Step 2: corners.
        var corners = Corners.IdentifyCorners(lat, lon, threshold: cornerThreshold);
        if (corners.Count == 0)
        {
            throw new InvalidOperationException("No corners detected. Try adjusting the corner detection threshold.");
        }

        var trackLength = refDist[^1];

        // Steps 3–5: per-lap arrays, zones, segments, stats.
        var throttleCh = channelNames["throttle"];
        var brakeCh = channelNames["brake"];
        var lateralCh = channelNames["lateral_g"];
        var speedCh = channelNames["gps_speed"];
        var wanted = new[] { "distance_m", throttleCh, brakeCh, lateralCh, "speed_kmh", speedCh }.Distinct().ToList();

        var lapArrays = new List<Zones.LapArrays>();
        var lapTimecodes = new List<long[]>();
        var lapLateral = new List<double[]>();
        foreach (var lap in selected)
        {
            cancellation.ThrowIfCancellationRequested();
            LogFile aligned;
            try
            {
                aligned = log.FilterByLap(lap.Num).SelectChannels(wanted).ResampleToChannel("distance_m");
            }
            catch (Exception e) when (e is KeyNotFoundException or ArgumentException)
            {
                continue;
            }

            var distance = aligned["distance_m"].Values;
            if (distance.Length == 0)
            {
                continue;
            }

            lapArrays.Add(new Zones.LapArrays(lap.Num, lap.LapTimeS, distance, aligned[brakeCh].Values, aligned[throttleCh].Values, aligned[speedCh].Values, aligned["speed_kmh"].Values));
            lapTimecodes.Add(aligned["distance_m"].Timecodes);
            lapLateral.Add(aligned[lateralCh].Values);
        }

        var (brakingZones, accelZones) = Zones.DetectZonesFromArrays(lapArrays);
        var segments = Zones.CreateTrackSegments(corners, brakingZones, accelZones, trackLength);
        var stats = Zones.ComputeSegmentStats(lapArrays, segments);

        // Per-corner aggregates from the stats.
        var cornerData = new List<CornerConsistency>();
        var traceBounds = new List<(double Start, double End)>();
        foreach (var corner in corners)
        {
            var cd = new CornerConsistency { Corner = corner };
            var bp = stats.Where(s => s.CornerId == corner.Id && s.SegmentType == "braking" && s.BrakingPoint is not null).Select(s => s.BrakingPoint!.Value).ToList();
            if (bp.Count > 0)
            {
                cd.BpValues.AddRange(bp);
                cd.BpStd = Numerics.Std(bp.ToArray());
            }

            var cornerStats = stats.Where(s => s.CornerId == corner.Id && s.SegmentType == "corner").ToList();
            var minSpeeds = cornerStats.Where(s => s.MinSpeed is not null).Select(s => s.MinSpeed!.Value).ToList();
            if (minSpeeds.Count > 0)
            {
                cd.SpeedValues.AddRange(minSpeeds);
                cd.SpeedMean = Numerics.Mean(minSpeeds.ToArray());
                cd.SpeedStd = Numerics.Std(minSpeeds.ToArray());
            }

            var exitSpeeds = cornerStats.Where(s => s.ExitSpeed is not null).Select(s => s.ExitSpeed!.Value).ToList();
            if (exitSpeeds.Count > 0)
            {
                cd.ExitSpeedValues.AddRange(exitSpeeds);
                cd.ExitSpeedMean = Numerics.Mean(exitSpeeds.ToArray());
                cd.ExitSpeedStd = Numerics.Std(exitSpeeds.ToArray());
            }

            var accelSeg = segments.FirstOrDefault(s => s.CornerId == corner.Id && s.SegmentType == "acceleration");
            if (accelSeg is not null)
            {
                cd.AccelZoneLength = accelSeg.Length;
                if (cd.ExitSpeedStd > 0)
                {
                    cd.OpportunityScore = cd.ExitSpeedStd * cd.AccelZoneLength;
                }
            }

            var brakingSeg = segments.FirstOrDefault(s => s.CornerId == corner.Id && s.SegmentType == "braking");
            double traceStart;
            if (brakingSeg is not null)
            {
                traceStart = brakingSeg.StartDist - 50.0;
                cd.BrakingStart = brakingSeg.StartDist;
            }
            else
            {
                traceStart = corner.StartDist - 150.0;
            }

            traceBounds.Add((traceStart, corner.EndDist + 50.0));
            cornerData.Add(cd);
        }

        // Steps 6–7: throttle acceptance and traces per lap and corner.
        for (var i = 0; i < lapArrays.Count; i++)
        {
            var lap = lapArrays[i];
            var timecodes = lapTimecodes[i];
            var lateral = lapLateral[i];
            var (smoothed, effective) = DriverAnalysis.PrepareThrottleAcceptance(lap.Throttle, lateral, throttleThreshold);
            for (var ci = 0; ci < corners.Count; ci++)
            {
                var (traceStart, traceEnd) = traceBounds[ci];
                var si = Numerics.SearchSortedLeft(lap.Distance, traceStart);
                var ei = Numerics.SearchSortedRight(lap.Distance, traceEnd);
                if (si >= ei)
                {
                    continue;
                }

                cornerData[ci].LapTraces.Add(new LapTrace(lap.LapNum, lap.Distance[si..ei], lap.Throttle[si..ei], lap.Brake[si..ei], lateral[si..ei]));
                var ta = DriverAnalysis.FindThrottleAcceptance(lap.Distance, timecodes, lap.Throttle, lateral, corners[ci], smoothed, effective, sustainTimeMs: sustainTimeMs);
                if (ta is not null)
                {
                    cornerData[ci].TaValues.Add(ta.ThrottleAcceptancePct);
                }
            }
        }

        foreach (var cd in cornerData)
        {
            if (cd.TaValues.Count > 0)
            {
                cd.TaMean = Numerics.Mean(cd.TaValues.ToArray());
                cd.TaStd = Numerics.Std(cd.TaValues.ToArray());
            }
        }

        return new DriverConsistencyResult(corners, cornerData, segments, stats, lat, lon, refDist);
    }
}
