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

    /// <summary>The lap the regions were listed on, as latitudes (with <see cref="RefLon"/>), for <see cref="Covers"/>.</summary>
    public double[] RefLat { get; init; } = Array.Empty<double>();

    /// <summary>Its longitudes.</summary>
    public double[] RefLon { get; init; } = Array.Empty<double>();

    /// <summary>Where each ticked region lies on that lap: its first and last sample, and its marker (a corner's apex).</summary>
    public IReadOnlyList<(TrackRegionKind Kind, int First, int Last, int Marker)> Spans { get; init; } = Array.Empty<(TrackRegionKind, int, int, int)>();

    /// <summary>A corner whose apex is within this of a ticked corner's is that corner.</summary>
    public const double SameCornerM = 30;

    /// <summary>A point further than this from the reference lap is not on this track at all.</summary>
    private const double OffTrackM = 50;

    /// <summary>
    /// Whether a corner with its apex at <paramref name="apex"/> is one of the ticked regions'
    /// corners: it is a ticked corner (its apex is where that one's is; a ticked corner's stretch
    /// starts with its braking, which can reach back past the previous corner's apex, so being
    /// inside the stretch is not the test) or it lies in a ticked sector. True for whole laps.
    /// Matching by place holds for a corner list detected from other laps than the regions were.
    /// </summary>
    public bool CoversCorner(GeoPoint apex)
    {
        if (IsWholeLap)
        {
            return true;
        }

        if (Spans.Any(s => s.Kind == TrackRegionKind.Corner && GeoProjection.DistanceM(new GeoPoint(RefLat[s.Marker], RefLon[s.Marker]), apex) <= SameCornerM))
        {
            return true;
        }

        var nearest = -1;
        var distance = double.MaxValue;
        for (var i = 0; i < RefLat.Length; i++)
        {
            var d = GeoProjection.DistanceM(new GeoPoint(RefLat[i], RefLon[i]), apex);
            if (d < distance)
            {
                distance = d;
                nearest = i;
            }
        }

        return distance <= OffTrackM && Spans.Any(s => s.Kind == TrackRegionKind.Sector && nearest >= s.First && nearest <= s.Last);
    }

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
    public RegionItemViewModel(int index, TrackRegion region, int firstSample, int lastSample, int markerSample, string tip, Action changed, string direction = "", string speed = "")
    {
        Direction = direction;
        Speed = speed;
        Index = index;
        Region = region;
        FirstSample = firstSample;
        LastSample = lastSample;
        MarkerSample = markerSample;
        Tip = tip;
        _changed = changed;
    }

    /// <summary>A corner's direction for its row, "L" or "R"; empty for a sector.</summary>
    public string Direction { get; }

    /// <summary>A corner's speed at its slowest point for its row ("138 km/h"); empty where unknown and for a sector.</summary>
    public string Speed { get; }

    /// <summary>What the region is, in a line: "Turn 3 · right · 138 km/h at the slowest point".</summary>
    public string Tip { get; }

    /// <summary>Position in the list; the id the map reports and highlights.</summary>
    public int Index { get; }

    /// <summary>The region.</summary>
    public TrackRegion Region { get; }

    /// <summary>Display name.</summary>
    public string Name => Region.Name;

    /// <summary>The name in short: "T3" for "Turn 3", a sector's name as it is (the map numbers its corners with it).</summary>
    public string ShortName => Region.Kind == TrackRegionKind.Corner && Name.StartsWith("Turn ", StringComparison.Ordinal) ? "T" + Name["Turn ".Length..] : Name;

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
/// The regions of the session's track and which are ticked: the selector in
/// the sidebar, under the lap lists. One choice for every tab; what a tab
/// does with it is the tab's <see cref="RegionUse"/> (the statistical tabs
/// cover only the ticked regions; a tab that does not use them shows whole
/// laps and the selector says so). The list is a property of the
/// session and its track, not of the ticked laps: corners are those of the
/// Driver tab's analysis (its channels and thresholds, through the same
/// <see cref="DriverConsistencyCache"/>) over the laps the session starts
/// with ticked (<see cref="SessionViewModel.DefaultAnalysisLaps"/>), and
/// sectors come from the saved track at this circuit.
/// </summary>
public sealed partial class RegionsViewModel : ObservableObject
{
    private readonly SessionViewModel _a;
    private readonly TrackStore _store;
    private readonly Func<SessionSlot, IReadOnlyDictionary<string, string>> _channelNames;
    private readonly Func<List<int>, DriverViewModel.SessionInputs?> _detectionInputs;
    private readonly DriverConsistencyCache _cache;
    private double[] _x = Array.Empty<double>();
    private double[] _y = Array.Empty<double>();
    private double[] _refLat = Array.Empty<double>();
    private double[] _refLon = Array.Empty<double>();
    private int _version;
    private int _requests;
    private bool _batch;
    private LogFile? _listedFor;

    /// <summary>
    /// Create the list for Session A. <paramref name="channelNames"/> gives a session's
    /// driver-analysis channel names (for the GPS channels laps are cut with);
    /// <paramref name="detectionInputs"/> gives what the Driver tab's analysis of Session A would
    /// read for some laps (<see cref="DriverViewModel.DetectionInputs"/>), run through <paramref name="cache"/>.
    /// </summary>
    public RegionsViewModel(
        SessionViewModel a,
        TrackStore store,
        Func<SessionSlot, IReadOnlyDictionary<string, string>> channelNames,
        Func<List<int>, DriverViewModel.SessionInputs?> detectionInputs,
        DriverConsistencyCache cache)
    {
        _a = a;
        _store = store;
        _channelNames = channelNames;
        _detectionInputs = detectionInputs;
        _cache = cache;
    }

    /// <summary>Detected corners.</summary>
    public ObservableCollection<RegionItemViewModel> Corners { get; } = new();

    /// <summary>Sectors of the saved track here.</summary>
    public ObservableCollection<RegionItemViewModel> Sectors { get; } = new();

    /// <summary>Every row, corners first; a row's <see cref="RegionItemViewModel.Index"/> is its position here.</summary>
    public IEnumerable<RegionItemViewModel> Items => Corners.Concat(Sectors);

    /// <summary>Raised when what the analyses should cover changed (a tick, or the reference toggle).</summary>
    public event Action? Changed;

    /// <summary>Draw the whole-lap result behind a region's, for reference (one setting for the tabs that can).</summary>
    [ObservableProperty]
    private bool _showWholeLap = true;

    /// <summary>What the tab now showing does with the ticks.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsUsed))]
    [NotifyPropertyChangedFor(nameof(Caption))]
    private RegionUse _use = RegionUse.Filter;

    /// <summary>The tab now showing uses the ticks; otherwise the selector is dimmed and the ticks wait for a tab that does.</summary>
    public bool IsUsed => Use != RegionUse.None;

    /// <summary>What the ticks do on the tab now showing, where that needs saying.</summary>
    public string Caption => Use == RegionUse.None ? "Not used on this tab: it shows whole laps" : string.Empty;

    /// <summary>The small map: the reference lap with the ticked regions coloured.</summary>
    [ObservableProperty]
    private TrackMapSpec? _mapSpec;

    /// <summary>
    /// The map's position marker: its one member is the map, and <see cref="ShowPosition"/> moves
    /// the marker round the lap.
    /// </summary>
    public CursorLinkGroup Position { get; } = new();

    /// <summary>
    /// Show on the map where something on the active tab is on the track (the cursor on the
    /// traces, a hovered corner): <paramref name="lapFraction"/> of the way round the lap from the
    /// start/finish line, or null to take the marker away.
    /// </summary>
    public void ShowPosition(double? lapFraction) => Position.SetCursor(lapFraction);

    /// <summary>Index of the row under the pointer, or -1; the map emphasises it.</summary>
    [ObservableProperty]
    private int _hoveredRegion = -1;

    /// <summary>What the list is waiting for or missing, or empty.</summary>
    [ObservableProperty]
    private string _hint = "Load a session to pick regions";

    /// <summary>Why there are no sectors, or empty when there are.</summary>
    [ObservableProperty]
    private string _sectorsHint = string.Empty;

    /// <summary>Session A is loaded: there is a track to pick regions of (the selector is hidden otherwise).</summary>
    [ObservableProperty]
    private bool _hasSession;

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
        var ticked = Items.Where(i => i.IsSelected).ToList();
        return new RegionSelection(
            ticked.Select(i => i.Region).ToList(),
            Label(),
            ShowWholeLap,
            (a["gps_latitude"], a["gps_longitude"]),
            (b["gps_latitude"], b["gps_longitude"]))
        {
            RefLat = _refLat,
            RefLon = _refLon,
            Spans = ticked.Select(i => (i.Region.Kind, i.FirstSample, i.LastSample, i.MarkerSample)).ToList(),
        };
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

    /// <summary>Tick the region at <paramref name="index"/> and nothing else (a double click on its row).</summary>
    public void Solo(int index)
    {
        _batch = true;
        foreach (var item in Items)
        {
            item.IsSelected = item.Index == index;
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

    /// <summary>One region as listed: where it lies on the reference lap (sample indices) and its tooltip.</summary>
    private sealed record Listed(TrackRegion Region, int First, int Last, int Marker, string Tip, string Direction = "", string Speed = "");

    /// <summary>What <see cref="Detect"/> found.</summary>
    private sealed record Detected(double[] Lat, double[] Lon, IReadOnlyList<Listed> Corners, IReadOnlyList<Listed> Sectors, string Hint, string SectorsHint);

    /// <summary>The refresh last started (completed when none has been): what a caller that did not start it can wait for.</summary>
    public Task LastRefresh { get; private set; } = Task.CompletedTask;

    /// <summary>
    /// Rebuild the list for the session now loaded: detect its corners (on a
    /// worker thread) and look up its track's sectors. Ticks do not carry over
    /// to another session. For the same session again (its track's gates or
    /// the Driver tab's thresholds were edited) the list stays up until the
    /// new one is known, nothing happens when that is the same list, and
    /// otherwise a tick is kept where its region is unchanged or, for a
    /// corner, where there is still a corner at the same place. A result for
    /// a session that has since been replaced is dropped.
    /// </summary>
    public Task RefreshAsync() => LastRefresh = RefreshCoreAsync();

    /// <summary>How long <see cref="RefreshSoonAsync"/> waits for further changes.</summary>
    public TimeSpan Debounce { get; set; } = TimeSpan.FromMilliseconds(300);

    /// <summary>
    /// <see cref="RefreshAsync"/> after <see cref="Debounce"/>, once for a burst of calls: for
    /// changes that arrive a keystroke at a time (a threshold or a channel name being typed).
    /// </summary>
    public Task RefreshSoonAsync() => LastRefresh = RefreshSoonCoreAsync();

    private async Task RefreshSoonCoreAsync()
    {
        var request = ++_requests;
        await Task.Delay(Debounce);
        if (request == _requests)
        {
            await RefreshCoreAsync();
        }
    }

    private async Task RefreshCoreAsync()
    {
        var version = ++_version;
        var log = _a.Log;
        var sameSession = log is not null && ReferenceEquals(log, _listedFor);
        HasSession = log is not null;
        if (!sameSession)
        {
            var hadSelection = HasSelection;
            _listedFor = null;
            Reset(log is null ? "Load a session to pick regions" : "Finding the corners…");
            if (hadSelection)
            {
                Changed?.Invoke();
            }
        }

        if (log is null)
        {
            return;
        }

        var names = _channelNames(SessionSlot.A);
        var gps = (names["gps_latitude"], names["gps_longitude"]);
        var laps = SessionViewModel.DefaultAnalysisLaps(log);
        var inputs = _detectionInputs(laps);
        var store = _store;
        var cache = _cache;
        var detected = await Task.Run(() => Detect(log, laps, gps, inputs, cache, store));
        if (version != _version)
        {
            return;
        }

        var listed = detected.Corners.Concat(detected.Sectors).ToList();
        if (sameSession && Hint == detected.Hint && SectorsHint == detected.SectorsHint && listed.Select(l => l.Region).SequenceEqual(Items.Select(i => i.Region)))
        {
            return; // the same list: nothing to redraw, nothing to recompute
        }

        var hadTicks = HasSelection;
        var keptRegions = Items.Where(i => i.IsSelected).Select(i => i.Region).ToList();
        var keptApexes = Items.Where(i => i.IsSelected && i.Region.Kind == TrackRegionKind.Corner)
            .Select(i => new GeoPoint(_refLat[i.MarkerSample], _refLon[i.MarkerSample])).ToList();
        Reset(detected.Hint);
        _refLat = detected.Lat;
        _refLon = detected.Lon;
        (_x, _y) = detected.Lat.Length > 0 ? Core.Analysis.Corners.GpsToLocalXy(detected.Lat, detected.Lon) : (Array.Empty<double>(), Array.Empty<double>());
        var index = 0;
        foreach (var l in detected.Corners)
        {
            Corners.Add(new RegionItemViewModel(index++, l.Region, l.First, l.Last, l.Marker, l.Tip, OnTicksChanged, l.Direction, l.Speed));
        }

        foreach (var l in detected.Sectors)
        {
            Sectors.Add(new RegionItemViewModel(index++, l.Region, l.First, l.Last, l.Marker, l.Tip, OnTicksChanged));
        }

        SectorsHint = detected.SectorsHint;
        OnPropertyChanged(nameof(HasCorners));
        OnPropertyChanged(nameof(HasSectors));
        _listedFor = log;
        _batch = true;
        foreach (var item in Items)
        {
            item.IsSelected = keptRegions.Contains(item.Region)
                || (item.Region.Kind == TrackRegionKind.Corner
                    && keptApexes.Any(apex => GeoProjection.DistanceM(apex, new GeoPoint(_refLat[item.MarkerSample], _refLon[item.MarkerSample])) <= RegionSelection.SameCornerM));
        }

        _batch = false;
        if (hadTicks)
        {
            OnTicksChanged(); // the kept ticks, or none: either way what the tabs cover may have changed
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

    private static Detected Detect(
        LogFile log, List<int> laps, (string Lat, string Lon) gps, DriverViewModel.SessionInputs? inputs, DriverConsistencyCache cache, TrackStore store)
    {
        var none = Array.Empty<Listed>();
        if (laps.Count == 0 || !log.TryGetChannel(gps.Lat, out _) || !log.TryGetChannel(gps.Lon, out _))
        {
            return new Detected(Array.Empty<double>(), Array.Empty<double>(), none, none, "Regions need GPS channels and at least one timed lap", string.Empty);
        }

        var corners = new List<Listed>();
        var hint = string.Empty;
        DriverConsistencyResult? result = null;
        if (inputs is null)
        {
            hint = "No corners: a Driver Consistency threshold is not valid (Channels…)";
        }
        else
        {
            try
            {
                result = DriverViewModel.Run(inputs, cache, CancellationToken.None).Result;
            }
            catch (Exception e) when (e is InvalidOperationException or KeyNotFoundException)
            {
                // The kernel's two "cannot" answers: no corners detected, or a pedal channel that is not in the file.
                hint = "No corners found: check the throttle, brake and GPS channels (Channels…)";
            }
        }

        double[] lat;
        double[] lon;
        if (result is not null)
        {
            lat = result.RefLat;
            lon = result.RefLon;
            foreach (var region in TrackRegions.FromCorners(result))
            {
                var data = result.CornerData.First(c => c.Corner.Name == region.Name);
                var segments = result.Segments.Where(s => s.CornerId == data.Corner.Id).ToList();
                var (first, last) = TrackRegions.IndexRange(result.RefDistance, segments.Min(s => s.StartDist), segments.Max(s => s.EndDist));
                var apexDist = segments.FirstOrDefault(s => s.SegmentType == "corner")?.ApexDist ?? (segments.Min(s => s.StartDist) + segments.Max(s => s.EndDist)) / 2;
                var apex = Numerics.ArgMin(result.RefDistance.Select(d => Math.Abs(d - apexDist)).ToArray());
                var tip = $"{region.Name} · {(data.Corner.Direction == 'L' ? "left" : "right")}";
                var speed = string.Empty;
                if (data.SpeedValues.Count > 0)
                {
                    speed = $"{data.SpeedMean.ToString("0", System.Globalization.CultureInfo.InvariantCulture)} km/h";
                    tip += $" · {speed} at the slowest point";
                }

                corners.Add(new Listed(region, first, last, apex, tip, data.Corner.Direction.ToString(), speed));
            }
        }
        else
        {
            // No corners: sectors still work off the fastest lap's path.
            var best = log.Laps.Where(l => laps.Contains(l.Num)).MinBy(l => l.LapTimeMs)!.Num;
            var path = log.FilterByLap(best, new[] { gps.Lat, gps.Lon }).ResampleToChannel(gps.Lat);
            lat = path[gps.Lat].Values;
            lon = path[gps.Lon].Values;
        }

        var sectors = new List<Listed>();
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
                    sectors.Add(new Listed(regions[s], first, last, (first + last) / 2, $"Sector {s + 1} of {track.Name}"));
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
                paths.Add(new TrackMapPathSpec(xs, ys, ChartColors.Region, 4));
            }

            // Sectors tile the lap, so only corners get a dot to click.
            if (item.Region.Kind == TrackRegionKind.Corner)
            {
                points.Add(new TrackMapPointSpec(_x[item.MarkerSample], _y[item.MarkerSample], ChartColors.Ink, 5, Label: item.ShortName[1..], Id: item.Index));
            }

            highlights.Add(new TrackMapHighlightSpec(item.Index, xs, ys, ChartColors.HighlightA, 5, _x[item.MarkerSample], _y[item.MarkerSample]));
        }

        // How far round the lap each sample of the outline is, for the position marker.
        var along = new double[_x.Length];
        for (var i = 1; i < along.Length; i++)
        {
            along[i] = along[i - 1] + Math.Sqrt(((_x[i] - _x[i - 1]) * (_x[i] - _x[i - 1])) + ((_y[i] - _y[i - 1]) * (_y[i] - _y[i - 1])));
        }

        var length = along[^1];
        for (var i = 0; i < along.Length && length > 0; i++)
        {
            along[i] /= length;
        }

        MapSpec = new TrackMapSpec(string.Empty, paths, points, highlights, Compact: true)
        {
            Cursor = new TrackMapCursorSpec(along, _x, _y, ChartColors.HighlightA, 11),
        };
    }
}
