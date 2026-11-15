using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>
/// What an analysis is restricted to, snapshotted for a worker thread: the
/// ticked regions (their union; none means whole laps), the GPS channels to
/// cut each session's laps with, and whether the whole-lap result is wanted
/// alongside for reference.
/// </summary>
/// <param name="Regions">The ticked regions.</param>
/// <param name="Label">Their names, for chart titles ("Turn 3 + S2").</param>
/// <param name="ShowWholeLap">Draw the whole-lap result behind the regions' as a reference.</param>
/// <param name="GpsA">Session A's latitude and longitude channels.</param>
/// <param name="GpsB">Session B's.</param>
public sealed record RegionSelection(
    IReadOnlyList<TrackRegion> Regions,
    string Label,
    bool ShowWholeLap,
    (string Lat, string Lon) GpsA,
    (string Lat, string Lon) GpsB)
{
    /// <summary>No restriction.</summary>
    public static RegionSelection WholeLap { get; } = new(Array.Empty<TrackRegion>(), string.Empty, false, ("GPS Latitude", "GPS Longitude"), ("GPS Latitude", "GPS Longitude"));

    /// <summary>Nothing is ticked: analyses cover whole laps.</summary>
    public bool IsWholeLap => Regions.Count == 0;

    /// <summary>
    /// Per lap, when the car was inside the ticked regions (null for whole
    /// laps: no cutting). Call on the worker thread.
    /// </summary>
    public IReadOnlyDictionary<int, IReadOnlyList<(long Start, long End)>>? Windows(LogFile log, IEnumerable<int> laps, SessionSlot slot)
    {
        if (IsWholeLap)
        {
            return null;
        }

        var (lat, lon) = slot == SessionSlot.A ? GpsA : GpsB;
        return TrackRegions.WindowsByLap(log, laps, Regions, lat, lon);
    }
}

/// <summary>One row of the region list.</summary>
public sealed partial class RegionItemViewModel : ObservableObject
{
    private readonly Action _changed;

    /// <summary>Create a row; <paramref name="changed"/> runs when its tick changes.</summary>
    public RegionItemViewModel(int index, TrackRegion region, int firstSample, int lastSample, int markerSample, Action changed)
    {
        Index = index;
        Region = region;
        FirstSample = firstSample;
        LastSample = lastSample;
        MarkerSample = markerSample;
        _changed = changed;
    }

    /// <summary>Position in the list; the id the map reports and highlights.</summary>
    public int Index { get; }

    /// <summary>The region.</summary>
    public TrackRegion Region { get; }

    /// <summary>Display name.</summary>
    public string Name => Region.Name;

    /// <summary>Where the region starts on the reference lap (sample index), for drawing.</summary>
    public int FirstSample { get; }

    /// <summary>Where it ends.</summary>
    public int LastSample { get; }

    /// <summary>Where its dot goes: the apex of a corner, the middle of a sector.</summary>
    public int MarkerSample { get; }

    /// <summary>Ticked: analyses are restricted to the ticked regions.</summary>
    [ObservableProperty]
    private bool _isSelected;

    partial void OnIsSelectedChanged(bool value) => _changed();
}

/// <summary>
/// The regions of the track the statistical tabs (Suspension, Roll Gradient,
/// Tire Grip) can be restricted to, and which are ticked. One choice shared
/// by those tabs, as their lap selection is. The list is a property of the
/// session and its track, not of the ticked laps: corners are detected once
/// per session from its laps within 103 % of the fastest, with the Driver
/// tab's channels, and sectors come from the saved track at this circuit.
/// </summary>
public sealed partial class RegionsViewModel : ObservableObject
{
    private readonly SessionViewModel _a;
    private readonly TrackStore _store;
    private readonly Func<SessionSlot, IReadOnlyDictionary<string, string>> _channelNames;
    private double[] _x = Array.Empty<double>();
    private double[] _y = Array.Empty<double>();
    private double[] _refLat = Array.Empty<double>();
    private double[] _refLon = Array.Empty<double>();
    private int _version;
    private bool _batch;
    private LogFile? _listedFor;

    /// <summary>Create the list for Session A; <paramref name="channelNames"/> gives a session's driver-analysis channel names.</summary>
    public RegionsViewModel(SessionViewModel a, TrackStore store, Func<SessionSlot, IReadOnlyDictionary<string, string>> channelNames)
    {
        _a = a;
        _store = store;
        _channelNames = channelNames;
    }

    /// <summary>Detected corners.</summary>
    public ObservableCollection<RegionItemViewModel> Corners { get; } = new();

    /// <summary>Sectors of the saved track here.</summary>
    public ObservableCollection<RegionItemViewModel> Sectors { get; } = new();

    /// <summary>Every row, corners first; a row's <see cref="RegionItemViewModel.Index"/> is its position here.</summary>
    public IEnumerable<RegionItemViewModel> Items => Corners.Concat(Sectors);

    /// <summary>Raised when what the analyses should cover changed (a tick, or the reference toggle).</summary>
    public event Action? Changed;

    /// <summary>Draw the whole-lap result behind a region's, for reference.</summary>
    [ObservableProperty]
    private bool _showWholeLap = true;

    /// <summary>The small map: the reference lap with the ticked regions coloured.</summary>
    [ObservableProperty]
    private TrackMapSpec? _mapSpec;

    /// <summary>Index of the row under the pointer, or -1; the map emphasises it.</summary>
    [ObservableProperty]
    private int _hoveredRegion = -1;

    /// <summary>What the list is waiting for or missing, or empty.</summary>
    [ObservableProperty]
    private string _hint = "Load a session to pick regions";

    /// <summary>Why there are no sectors, or empty when there are.</summary>
    [ObservableProperty]
    private string _sectorsHint = string.Empty;

    /// <summary>There are corners to list.</summary>
    public bool HasCorners => Corners.Count > 0;

    /// <summary>There are sectors to list.</summary>
    public bool HasSectors => Sectors.Count > 0;

    /// <summary>Something is ticked.</summary>
    public bool HasSelection => Items.Any(i => i.IsSelected);

    /// <summary>"Whole lap", or the ticked regions' names.</summary>
    public string Summary => Label().Length == 0 ? "Whole lap" : Label();

    partial void OnShowWholeLapChanged(bool value) => Changed?.Invoke();

    /// <summary>The current choice, for a worker thread.</summary>
    public RegionSelection Selection()
    {
        var a = _channelNames(SessionSlot.A);
        var b = _channelNames(SessionSlot.B);
        return new RegionSelection(
            Items.Where(i => i.IsSelected).Select(i => i.Region).ToList(),
            Label(),
            ShowWholeLap,
            (a["gps_latitude"], a["gps_longitude"]),
            (b["gps_latitude"], b["gps_longitude"]));
    }

    private string Label()
    {
        var names = Items.Where(i => i.IsSelected).Select(i => i.Name).ToList();
        return names.Count <= 3 ? string.Join(" + ", names) : $"{names.Count} regions";
    }

    /// <summary>Untick everything: back to whole laps.</summary>
    [RelayCommand]
    public void Clear()
    {
        _batch = true;
        foreach (var item in Items)
        {
            item.IsSelected = false;
        }

        _batch = false;
        OnTicksChanged();
    }

    /// <summary>Flip the tick of the row at <paramref name="index"/> (a click on its dot on the map).</summary>
    public void Toggle(int index)
    {
        if (Items.ElementAtOrDefault(index) is { } item)
        {
            item.IsSelected = !item.IsSelected;
        }
    }

    private void OnTicksChanged()
    {
        if (_batch)
        {
            return;
        }

        BuildMap();
        OnPropertyChanged(nameof(HasSelection));
        OnPropertyChanged(nameof(Summary));
        Changed?.Invoke();
    }

    /// <summary>What <see cref="Detect"/> found.</summary>
    private sealed record Detected(
        double[] Lat,
        double[] Lon,
        IReadOnlyList<(TrackRegion Region, int First, int Last, int Marker)> Corners,
        IReadOnlyList<(TrackRegion Region, int First, int Last, int Marker)> Sectors,
        string Hint,
        string SectorsHint);

    /// <summary>The refresh last started (completed when none has been): what a caller that did not start it can wait for.</summary>
    public Task LastRefresh { get; private set; } = Task.CompletedTask;

    /// <summary>
    /// Rebuild the list for the session now loaded: detect its corners (on a
    /// worker thread) and look up its track's sectors. Ticks do not carry over
    /// to another session. For the same session again (its track's gates were
    /// edited) the ticked corners stay ticked; sectors do not, since "S2" is a
    /// different stretch once a gate has moved. A result for a session that has
    /// since been replaced is dropped.
    /// </summary>
    public Task RefreshAsync() => LastRefresh = RefreshCoreAsync();

    private async Task RefreshCoreAsync()
    {
        var version = ++_version;
        var log = _a.Log;
        var hadSelection = HasSelection;
        var sameSession = log is not null && ReferenceEquals(log, _listedFor);
        var keptCorners = sameSession ? Corners.Where(c => c.IsSelected).Select(c => c.Region.Id).ToHashSet() : new HashSet<string>();
        _listedFor = null;
        Reset(log is null ? "Load a session to pick regions" : "Finding the corners…");
        if (hadSelection && !sameSession)
        {
            Changed?.Invoke();
        }

        if (log is null)
        {
            return;
        }

        var names = _channelNames(SessionSlot.A);
        var store = _store;
        var detected = await Task.Run(() => Detect(log, names, store));
        if (version != _version)
        {
            return;
        }

        _refLat = detected.Lat;
        _refLon = detected.Lon;
        (_x, _y) = detected.Lat.Length > 0 ? Core.Analysis.Corners.GpsToLocalXy(detected.Lat, detected.Lon) : (Array.Empty<double>(), Array.Empty<double>());
        var index = 0;
        foreach (var (region, first, last, marker) in detected.Corners)
        {
            Corners.Add(new RegionItemViewModel(index++, region, first, last, marker, OnTicksChanged));
        }

        foreach (var (region, first, last, marker) in detected.Sectors)
        {
            Sectors.Add(new RegionItemViewModel(index++, region, first, last, marker, OnTicksChanged));
        }

        Hint = detected.Hint;
        SectorsHint = detected.SectorsHint;
        OnPropertyChanged(nameof(HasCorners));
        OnPropertyChanged(nameof(HasSectors));
        _listedFor = log;
        _batch = true;
        foreach (var corner in Corners.Where(c => keptCorners.Contains(c.Region.Id)))
        {
            corner.IsSelected = true;
        }

        _batch = false;
        if (hadSelection && sameSession)
        {
            OnTicksChanged(); // the kept ticks, or none where only sectors were ticked: either way the tabs recompute
        }
        else
        {
            BuildMap();
        }
    }

    private void Reset(string hint)
    {
        Corners.Clear();
        Sectors.Clear();
        _x = Array.Empty<double>();
        _y = Array.Empty<double>();
        HoveredRegion = -1;
        MapSpec = null;
        Hint = hint;
        SectorsHint = string.Empty;
        OnPropertyChanged(nameof(HasCorners));
        OnPropertyChanged(nameof(HasSectors));
        OnPropertyChanged(nameof(HasSelection));
        OnPropertyChanged(nameof(Summary));
    }

    private static Detected Detect(LogFile log, IReadOnlyDictionary<string, string> names, TrackStore store)
    {
        var none = Array.Empty<(TrackRegion, int, int, int)>();
        var laps = LapSelection.GetTopLaps(log.Laps).Select(l => l.Num).ToList();
        if (laps.Count == 0 || !log.TryGetChannel(names["gps_latitude"], out _) || !log.TryGetChannel(names["gps_longitude"], out _))
        {
            return new Detected(Array.Empty<double>(), Array.Empty<double>(), none, none, "Regions need GPS channels and at least one timed lap", string.Empty);
        }

        double[] lat;
        double[] lon;
        var corners = new List<(TrackRegion, int, int, int)>();
        var hint = string.Empty;
        try
        {
            var threshold = DriverViewModel.AutoThrottleThreshold(log, names["throttle"]) ?? 98.0;
            var result = DriverConsistency.Analyze(log, laps, names, throttleThreshold: threshold);
            lat = result.RefLat;
            lon = result.RefLon;
            var regions = TrackRegions.FromCorners(result);
            foreach (var region in regions)
            {
                var corner = result.Corners.First(c => c.Name == region.Name);
                var segments = result.Segments.Where(s => s.CornerId == corner.Id).ToList();
                var (first, last) = TrackRegions.IndexRange(result.RefDistance, segments.Min(s => s.StartDist), segments.Max(s => s.EndDist));
                var apexDist = segments.FirstOrDefault(s => s.SegmentType == "corner")?.ApexDist ?? (segments.Min(s => s.StartDist) + segments.Max(s => s.EndDist)) / 2;
                var apex = Numerics.ArgMin(result.RefDistance.Select(d => Math.Abs(d - apexDist)).ToArray());
                corners.Add((region, first, last, apex));
            }
        }
        catch (Exception e) when (e is InvalidOperationException or KeyNotFoundException or ArgumentException)
        {
            // No corners (missing pedal channels, or none detected): sectors still work off the fastest lap's path.
            var best = LapSelection.GetBestLap(log.Laps)!.Num;
            var gps = log.FilterByLap(best, new[] { names["gps_latitude"], names["gps_longitude"] }).ResampleToChannel(names["gps_latitude"]);
            lat = gps[names["gps_latitude"]].Values;
            lon = gps[names["gps_longitude"]].Values;
            hint = "No corners found: check the throttle, brake and GPS channels (Channels…)";
        }

        var sectors = new List<(TrackRegion, int, int, int)>();
        var sectorsHint = "Add gates on the Sectors tab to pick sectors here";
        if (lat.Length > 2 && store.FindNear(new GeoPoint(lat[0], lon[0])) is { Gates.Count: > 0 } track)
        {
            // Where each gate sits on the reference lap: the sample nearest its middle.
            var gateSamples = track.Gates.Select(g => NearestSample(lat, lon, new GeoPoint((g.A.Lat + g.B.Lat) / 2, (g.A.Lon + g.B.Lon) / 2))).ToList();
            var regions = TrackRegions.FromSectors(track);
            for (var s = 0; s < regions.Count; s++)
            {
                var first = s == 0 ? 0 : gateSamples[s - 1];
                var last = s == regions.Count - 1 ? lat.Length - 1 : gateSamples[s];
                if (last > first)
                {
                    sectors.Add((regions[s], first, last, (first + last) / 2));
                }
            }

            sectorsHint = string.Empty;
        }

        return new Detected(lat, lon, corners, sectors, hint, sectorsHint);
    }

    private static int NearestSample(double[] lat, double[] lon, GeoPoint point)
    {
        var best = 0;
        var bestDistance = double.MaxValue;
        for (var i = 0; i < lat.Length; i++)
        {
            var d = GeoProjection.DistanceM(new GeoPoint(lat[i], lon[i]), point);
            if (d < bestDistance)
            {
                bestDistance = d;
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// The reference lap as an outline, the ticked regions coloured over it,
    /// a clickable dot per region, and a highlight per region for the row
    /// under the pointer.
    /// </summary>
    private void BuildMap()
    {
        if (_x.Length < 2)
        {
            MapSpec = null;
            return;
        }

        var paths = new List<TrackMapPathSpec> { new(_x, _y, ChartColors.Neutral, 2) };
        var points = new List<TrackMapPointSpec>();
        var highlights = new List<TrackMapHighlightSpec>();
        foreach (var item in Items)
        {
            var count = item.LastSample - item.FirstSample + 1;
            var xs = _x.AsSpan(item.FirstSample, count).ToArray();
            var ys = _y.AsSpan(item.FirstSample, count).ToArray();
            if (item.IsSelected)
            {
                paths.Add(new TrackMapPathSpec(xs, ys, ChartColors.SessionA, 4));
            }

            // Sectors tile the lap, so only corners get a dot to click.
            if (item.Region.Kind == TrackRegionKind.Corner)
            {
                points.Add(new TrackMapPointSpec(_x[item.MarkerSample], _y[item.MarkerSample], ChartColors.Ink, 5, Id: item.Index));
            }

            highlights.Add(new TrackMapHighlightSpec(item.Index, xs, ys, ChartColors.HighlightA, 5, _x[item.MarkerSample], _y[item.MarkerSample]));
        }

        MapSpec = new TrackMapSpec(string.Empty, paths, points, highlights, Compact: true);
    }
}
