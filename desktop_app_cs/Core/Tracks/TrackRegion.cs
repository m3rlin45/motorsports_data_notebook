using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Tracks;

/// <summary>What a <see cref="TrackRegion"/> was made from.</summary>
public enum TrackRegionKind
{
    /// <summary>A detected corner, from where its braking starts to where its acceleration ends.</summary>
    Corner,

    /// <summary>A sector of a saved track: the stretch between two of its gates.</summary>
    Sector,
}

/// <summary>
/// A stretch of track an analysis can be restricted to: everything a lap
/// drives between crossing <see cref="Start"/> and crossing <see cref="End"/>.
/// The gates are geographic, so the same region cuts every lap of every
/// session on that track where the car actually was, without relying on lap
/// distances agreeing between laps.
/// </summary>
/// <param name="Id">Stable within one list of regions ("corner:3", "sector:1").</param>
/// <param name="Name">Display name ("Turn 3", "S1").</param>
/// <param name="Kind">Corner or sector.</param>
/// <param name="Anchor">Origin of the local frame the gates are projected in.</param>
/// <param name="Start">Gate that opens the region; null for the start of the lap.</param>
/// <param name="End">Gate that closes it; null for the end of the lap.</param>
public sealed record TrackRegion(string Id, string Name, TrackRegionKind Kind, GeoPoint Anchor, SectorGate? Start, SectorGate? End);

/// <summary>Builds <see cref="TrackRegion"/>s and cuts laps with them.</summary>
public static class TrackRegions
{
    /// <summary>Half the length of a corner's gate chords: enough to span the track, short enough not to reach a neighbouring stretch.</summary>
    public const double CornerGateHalfWidthM = 20;

    /// <summary>
    /// One region per detected corner, in corner order: from the start of the
    /// earliest of its segments (braking, where there is any) to the end of
    /// the latest (acceleration), with gates across the reference lap there.
    /// A corner that runs up to an end of the reference lap is open at that end.
    /// </summary>
    public static IReadOnlyList<TrackRegion> FromCorners(DriverConsistencyResult result)
    {
        var regions = new List<TrackRegion>();
        var lat = result.RefLat;
        var lon = result.RefLon;
        var distance = result.RefDistance;
        if (lat.Length < 3)
        {
            return regions;
        }

        var anchor = new GeoPoint(lat[0], lon[0]);
        for (var c = 0; c < result.Corners.Count; c++)
        {
            var corner = result.Corners[c];
            var segments = result.Segments.Where(s => s.CornerId == corner.Id).ToList();
            if (segments.Count == 0)
            {
                continue;
            }

            var (first, last) = IndexRange(distance, segments.Min(s => s.StartDist), segments.Max(s => s.EndDist));
            if (last - first < 2)
            {
                continue;
            }

            var start = first <= 1 ? null : SectorTimes.GateAcrossPath(lat, lon, first, $"{corner.Name} start", anchor, CornerGateHalfWidthM);
            var end = last >= lat.Length - 2 ? null : SectorTimes.GateAcrossPath(lat, lon, last, $"{corner.Name} end", anchor, CornerGateHalfWidthM);
            regions.Add(new TrackRegion($"corner:{c}", corner.Name, TrackRegionKind.Corner, anchor, start, end));
        }

        return regions;
    }

    /// <summary>One region per sector of <paramref name="track"/> (gates + 1 of them, the first from the lap's start, the last to its end); none for a track without gates.</summary>
    public static IReadOnlyList<TrackRegion> FromSectors(TrackDefinition track)
    {
        var regions = new List<TrackRegion>();
        if (track.Gates.Count == 0)
        {
            return regions;
        }

        for (var s = 0; s <= track.Gates.Count; s++)
        {
            var start = s == 0 ? null : track.Gates[s - 1];
            var end = s == track.Gates.Count ? null : track.Gates[s];
            regions.Add(new TrackRegion($"sector:{s}", $"S{s + 1}", TrackRegionKind.Sector, track.Anchor, start, end));
        }

        return regions;
    }

    /// <summary>First and last index of <paramref name="distance"/> (ascending) within [<paramref name="from"/>, <paramref name="to"/>]; last &lt; first when there is none.</summary>
    public static (int First, int Last) IndexRange(ReadOnlySpan<double> distance, double from, double to)
    {
        var first = -1;
        var last = -2;
        for (var i = 0; i < distance.Length; i++)
        {
            if (distance[i] >= from && distance[i] <= to)
            {
                if (first < 0)
                {
                    first = i;
                }

                last = i;
            }
        }

        return (first, last);
    }

    /// <summary>
    /// When <paramref name="lap"/> was inside <paramref name="region"/>, on
    /// the session clock in ms: from crossing the start gate (or the lap's
    /// start) to the next crossing of the end gate (or the lap's end). Null
    /// when the lap's GPS path does not cross them in that order.
    /// </summary>
    /// <param name="region">The region.</param>
    /// <param name="lap">The lap.</param>
    /// <param name="lat">The lap's latitude samples.</param>
    /// <param name="lon">The lap's longitude samples.</param>
    /// <param name="timecodes">Their timecodes (ms on the session clock).</param>
    public static (long Start, long End)? Window(TrackRegion region, Lap lap, ReadOnlySpan<double> lat, ReadOnlySpan<double> lon, ReadOnlySpan<long> timecodes)
    {
        var gates = new List<SectorGate>(2);
        if (region.Start is not null)
        {
            gates.Add(region.Start);
        }

        if (region.End is not null)
        {
            gates.Add(region.End);
        }

        var crossings = gates.Count == 0
            ? Array.Empty<double>()
            : SectorTimes.GateCrossingsS(lat, lon, timecodes, new TrackDefinition(region.Id, region.Name, region.Anchor, gates), lap.StartTime);
        if (crossings.Any(double.IsNaN))
        {
            return null;
        }

        var next = 0;
        var start = region.Start is null ? lap.StartTime : lap.StartTime + (long)Math.Round(crossings[next++] * 1000.0);
        var end = region.End is null ? lap.EndTime : lap.StartTime + (long)Math.Round(crossings[next] * 1000.0);
        return end > start ? (start, end) : null;
    }

    /// <summary>
    /// The windows of every region in <paramref name="regions"/> on one lap,
    /// overlapping ones merged, in time order: the union a selection of
    /// several regions stands for. Regions the lap does not drive through
    /// contribute nothing.
    /// </summary>
    public static IReadOnlyList<(long Start, long End)> Windows(IEnumerable<TrackRegion> regions, Lap lap, ReadOnlySpan<double> lat, ReadOnlySpan<double> lon, ReadOnlySpan<long> timecodes)
    {
        var windows = new List<(long Start, long End)>();
        foreach (var region in regions)
        {
            if (Window(region, lap, lat, lon, timecodes) is { } window)
            {
                windows.Add(window);
            }
        }

        return Merge(windows);
    }

    /// <summary>
    /// Per lap of <paramref name="log"/> in <paramref name="lapNumbers"/>, the merged windows of
    /// <paramref name="regions"/> (see <see cref="Windows(IEnumerable{TrackRegion}, Lap, ReadOnlySpan{double}, ReadOnlySpan{double}, ReadOnlySpan{long})"/>).
    /// </summary>
    public static IReadOnlyDictionary<int, IReadOnlyList<(long Start, long End)>> WindowsByLap(
        LogFile log, IEnumerable<int> lapNumbers, IReadOnlyList<TrackRegion> regions, string latChannel, string lonChannel)
    {
        var result = new Dictionary<int, IReadOnlyList<(long Start, long End)>>();
        foreach (var num in lapNumbers)
        {
            var lap = log.Laps.FirstOrDefault(l => l.Num == num);
            if (lap is null)
            {
                continue;
            }

            var gps = log.FilterByLap(num, new[] { latChannel, lonChannel }).ResampleToChannel(latChannel);
            result[num] = Windows(regions, lap, gps[latChannel].Values, gps[lonChannel].Values, gps[latChannel].Timecodes);
        }

        return result;
    }

    /// <summary>Sort and merge overlapping or touching windows.</summary>
    public static IReadOnlyList<(long Start, long End)> Merge(IEnumerable<(long Start, long End)> windows)
    {
        var merged = new List<(long Start, long End)>();
        foreach (var window in windows.OrderBy(w => w.Start))
        {
            if (merged.Count > 0 && window.Start <= merged[^1].End)
            {
                merged[^1] = (merged[^1].Start, Math.Max(merged[^1].End, window.End));
            }
            else
            {
                merged.Add(window);
            }
        }

        return merged;
    }

    /// <summary>Whether <paramref name="timecode"/> lies in one of <paramref name="windows"/> (sorted, disjoint; ends inclusive).</summary>
    public static bool Contains(IReadOnlyList<(long Start, long End)> windows, long timecode)
    {
        foreach (var (start, end) in windows)
        {
            if (timecode < start)
            {
                return false;
            }

            if (timecode <= end)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Indices of the <paramref name="timecodes"/> that lie inside <paramref name="windows"/> (see <see cref="Contains"/>).</summary>
    public static int[] Keep(IReadOnlyList<(long Start, long End)> windows, ReadOnlySpan<long> timecodes)
    {
        var keep = new List<int>();
        for (var i = 0; i < timecodes.Length; i++)
        {
            if (Contains(windows, timecodes[i]))
            {
                keep.Add(i);
            }
        }

        return keep.ToArray();
    }

    /// <summary>The elements of <paramref name="source"/> at <paramref name="indices"/>.</summary>
    public static T[] Take<T>(T[] source, int[] indices)
    {
        var result = new T[indices.Length];
        for (var i = 0; i < indices.Length; i++)
        {
            result[i] = source[indices[i]];
        }

        return result;
    }

    /// <summary>Total length of <paramref name="windows"/> in seconds.</summary>
    public static double DurationS(IEnumerable<(long Start, long End)> windows) => windows.Sum(w => w.End - w.Start) / 1000.0;
}
