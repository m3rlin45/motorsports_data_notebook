using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>A gate being edited.</summary>
public sealed partial class GateViewModel : ObservableObject
{
    /// <summary>Create from a gate.</summary>
    public GateViewModel(SectorGate gate)
    {
        _name = gate.Name;
        A = gate.A;
        B = gate.B;
    }

    [ObservableProperty]
    private string _name;

    /// <summary>One end of the chord.</summary>
    public GeoPoint A { get; set; }

    /// <summary>The other end.</summary>
    public GeoPoint B { get; set; }

    /// <summary>The gate.</summary>
    public SectorGate ToGate() => new(Name, A, B);
}

/// <summary>
/// The Sectors tab: pick or create a track for the session, place sector
/// gates on the map, and see the split times of the selected laps.
/// </summary>
public sealed partial class SectorsViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _a;
    private readonly TrackStore _store;
    private bool _suppressTrackChange;

    /// <summary>Create the tab.</summary>
    public SectorsViewModel(SessionViewModel a, TrackStore store)
    {
        _a = a;
        _store = store;
        Tracks = new ObservableCollection<TrackDefinition>(_store.LoadAll());
    }

    /// <inheritdoc />
    public override string Header => "Sectors";

    /// <summary>Known tracks.</summary>
    public ObservableCollection<TrackDefinition> Tracks { get; }

    /// <summary>The track being used/edited.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasTrack))]
    [NotifyPropertyChangedFor(nameof(TrackName))]
    private TrackDefinition? _selectedTrack;

    /// <summary>Whether a track is selected.</summary>
    public bool HasTrack => SelectedTrack is not null;

    /// <summary>Editable name of the selected track.</summary>
    public string TrackName
    {
        get => SelectedTrack?.Name ?? string.Empty;
        set
        {
            if (SelectedTrack is { } t && value != t.Name)
            {
                SelectedTrack = t with { Name = value };
            }
        }
    }

    /// <summary>Gates of the selected track, editable.</summary>
    public ObservableCollection<GateViewModel> Gates { get; } = new();

    /// <summary>The gate selected in the editor.</summary>
    [ObservableProperty]
    private GateViewModel? _selectedGate;

    /// <summary>Reference lap path for the editor (local metres around the track anchor).</summary>
    [ObservableProperty]
    private (double[] X, double[] Y)? _referencePath;

    /// <summary>Reference lap GPS samples, index-aligned with <see cref="ReferencePath"/>.</summary>
    private double[] _refLat = Array.Empty<double>();
    private double[] _refLon = Array.Empty<double>();

    /// <summary>The split table.</summary>
    [ObservableProperty]
    private SplitTable? _table;

    /// <summary>Lap highlighted in the table.</summary>
    [ObservableProperty]
    private int? _highlightedLap;

    /// <summary>Whether the track differs from what is stored.</summary>
    [ObservableProperty]
    private bool _isDirty;

    partial void OnSelectedTrackChanged(TrackDefinition? value)
    {
        if (_suppressTrackChange)
        {
            return;
        }

        Gates.Clear();
        if (value is not null)
        {
            foreach (var g in value.Gates)
            {
                Gates.Add(new GateViewModel(g));
            }
        }

        SelectedGate = null;
        IsDirty = false;
        RebuildReferencePath();
        OnSelectionChanged();
    }

    private (string Lat, string Lon)? GpsChannels()
    {
        var names = _a.Profile?.ChannelNames ?? ChannelKeys.DefaultChannelNames;
        var lat = names["gps_latitude"];
        var lon = names["gps_longitude"];
        return _a.Log is not null && _a.Log.HasChannel(lat) && _a.Log.HasChannel(lon) ? (lat, lon) : null;
    }

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        if (_a.Log is not null && GpsChannels() is { } gps && SelectedTrack is null)
        {
            var first = FirstFix(_a.Log, gps);
            if (first is { } p)
            {
                SelectedTrack = _store.FindNear(p);
            }
        }

        RebuildReferencePath();
        base.OnSessionLoaded();
    }

    private static GeoPoint? FirstFix(LogFile log, (string Lat, string Lon) gps)
    {
        var lat = log[gps.Lat].Values;
        var lon = log[gps.Lon].Values;
        for (var i = 0; i < Math.Min(lat.Length, lon.Length); i++)
        {
            if (lat[i] != 0 || lon[i] != 0)
            {
                return new GeoPoint(lat[i], lon[i]);
            }
        }

        return null;
    }

    private void RebuildReferencePath()
    {
        ReferencePath = null;
        if (_a.Log is null || GpsChannels() is not { } gps)
        {
            return;
        }

        var laps = _a.SelectedLaps;
        var lapNum = laps.Count > 0 ? laps.OrderBy(n => _a.Log.Laps.First(l => l.Num == n).LapTimeMs).First() : (_a.Log.Laps.FirstOrDefault()?.Num ?? -1);
        if (lapNum < 0)
        {
            return;
        }

        var lap = _a.Log.FilterByLap(lapNum).SelectChannels(new[] { gps.Lat, gps.Lon }).ResampleToChannel(gps.Lat);
        _refLat = lap[gps.Lat].Values;
        _refLon = lap[gps.Lon].Values;
        var origin = SelectedTrack?.Anchor ?? new GeoPoint(_refLat.Length > 0 ? _refLat[0] : 0, _refLon.Length > 0 ? _refLon[0] : 0);
        var x = new double[_refLat.Length];
        var y = new double[_refLat.Length];
        for (var i = 0; i < x.Length; i++)
        {
            (x[i], y[i]) = GeoProjection.ToLocal(new GeoPoint(_refLat[i], _refLon[i]), origin);
        }

        ReferencePath = (x, y);
    }

    /// <summary>Local coordinates of a gate against the current anchor.</summary>
    public ((double X, double Y) A, (double X, double Y) B)? GateLocal(GateViewModel gate)
    {
        if (SelectedTrack is null)
        {
            return null;
        }

        return (GeoProjection.ToLocal(gate.A, SelectedTrack.Anchor), GeoProjection.ToLocal(gate.B, SelectedTrack.Anchor));
    }

    /// <summary>Create a new track anchored at the session's first GPS fix and select it.</summary>
    [RelayCommand]
    public void NewTrack()
    {
        if (_a.Log is null || GpsChannels() is not { } gps || FirstFix(_a.Log, gps) is not { } anchor)
        {
            Status = "Load a session with GPS to create a track";
            return;
        }

        var name = _a.Log.Metadata.TryGetValue("Venue", out var venue) && venue.Length > 0 ? venue : "New track";
        var id = TrackStore.IdFor(name);
        if (Tracks.Any(t => t.Id == id))
        {
            id += "_" + DateTime.Now.ToString("yyyyMMddHHmm");
        }

        var track = new TrackDefinition(id, name, anchor, Array.Empty<SectorGate>());
        Tracks.Add(track);
        SelectedTrack = track;
        IsDirty = true;
        Status = $"New track '{name}' — click the map to add gates, then Save";
    }

    /// <summary>Add a gate across the reference path at sample <paramref name="index"/>.</summary>
    public void AddGateAt(int index)
    {
        if (SelectedTrack is null || _refLat.Length == 0)
        {
            return;
        }

        index = Math.Clamp(index, 0, _refLat.Length - 1);
        var gate = SectorTimes.GateAcrossPath(_refLat, _refLon, index, $"Gate {Gates.Count + 1}", SelectedTrack.Anchor);
        var vm = new GateViewModel(gate);
        // Keep gates in lap order by the sample they sit at.
        var position = GateOrder().Count(i => i < index);
        Gates.Insert(position, vm);
        SelectedGate = vm;
        TouchTrack();
    }

    /// <summary>Move the selected gate to the path sample at <paramref name="index"/>.</summary>
    public void MoveSelectedGateTo(int index)
    {
        if (SelectedGate is null || SelectedTrack is null || _refLat.Length == 0)
        {
            return;
        }

        index = Math.Clamp(index, 0, _refLat.Length - 1);
        var gate = SectorTimes.GateAcrossPath(_refLat, _refLon, index, SelectedGate.Name, SelectedTrack.Anchor);
        SelectedGate.A = gate.A;
        SelectedGate.B = gate.B;
        var selected = SelectedGate;
        var ordered = Gates.OrderBy(g => NearestIndex(g)).ToList();
        Gates.Clear();
        foreach (var g in ordered)
        {
            Gates.Add(g);
        }

        SelectedGate = selected;
        TouchTrack();
    }

    private IEnumerable<int> GateOrder() => Gates.Select(NearestIndex);

    /// <summary>Index of the reference-path sample nearest to a gate's midpoint.</summary>
    public int NearestIndex(GateViewModel gate)
    {
        if (ReferencePath is not { } path || GateLocal(gate) is not { } local)
        {
            return 0;
        }

        var mx = (local.A.X + local.B.X) / 2;
        var my = (local.A.Y + local.B.Y) / 2;
        var best = 0;
        var bestD2 = double.PositiveInfinity;
        for (var i = 0; i < path.X.Length; i++)
        {
            var dx = path.X[i] - mx;
            var dy = path.Y[i] - my;
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2)
            {
                bestD2 = d2;
                best = i;
            }
        }

        return best;
    }

    /// <summary>Remove the selected gate.</summary>
    [RelayCommand]
    public void DeleteGate()
    {
        if (SelectedGate is null)
        {
            return;
        }

        Gates.Remove(SelectedGate);
        SelectedGate = null;
        TouchTrack();
    }

    /// <summary>Store the track (user tracks file).</summary>
    [RelayCommand]
    public void SaveTrack()
    {
        if (SelectedTrack is null)
        {
            return;
        }

        var track = CurrentTrack();
        _store.Save(track);
        _suppressTrackChange = true;
        var index = Tracks.IndexOf(SelectedTrack);
        Tracks[index] = track;
        SelectedTrack = track;
        _suppressTrackChange = false;
        IsDirty = false;
        Status = $"Track '{track.Name}' saved with {track.Gates.Count} gates";
    }

    /// <summary>The selected track with the edited gates.</summary>
    public TrackDefinition CurrentTrack() =>
        SelectedTrack! with { Gates = Gates.Select(g => g.ToGate()).ToList() };

    private void TouchTrack()
    {
        IsDirty = true;
        OnSelectionChanged();
    }

    private sealed record Parameters(LogFile Log, List<int> Laps, TrackDefinition Track, string Lat, string Lon);

    /// <inheritdoc />
    protected override object? GetParameters()
    {
        if (_a.Log is null)
        {
            return null;
        }

        if (GpsChannels() is not { } gps)
        {
            Status = "No GPS channels in the session";
            return null;
        }

        if (SelectedTrack is null)
        {
            Status = "Select a track or create one for this session";
            return null;
        }

        var laps = _a.SelectedLaps;
        if (laps.Count == 0)
        {
            Status = "Select laps to analyze";
            return null;
        }

        return new Parameters(_a.Log, laps, CurrentTrack(), gps.Lat, gps.Lon);
    }

    private SplitTable? _pending;

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        _pending = SectorTimes.Build(p.Log, p.Laps, p.Track, p.Lat, p.Lon);
    }

    /// <inheritdoc />
    protected override void Display()
    {
        Table = _pending;
        if (Table is { } t)
        {
            var incomplete = t.Laps.Count(l => !l.Complete);
            Status = incomplete == 0
                ? $"{t.Laps.Count} laps, {t.Track.SectorCount} sectors — theoretical best {LapItemViewModel.FormatLapTime(t.TheoreticalBestS)}"
                : $"{incomplete} of {t.Laps.Count} laps miss a gate — move the gates onto the racing line";
        }
    }
}
