using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>A gate being edited.</summary>
public sealed partial class GateViewModel : ObservableObject
{
    private readonly Action? _onRenamed;

    /// <summary>Create from a gate; <paramref name="onRenamed"/> is called when the user edits the name.</summary>
    public GateViewModel(SectorGate gate, Action? onRenamed = null)
    {
        _name = gate.Name;
        A = gate.A;
        B = gate.B;
        _onRenamed = onRenamed;
    }

    [ObservableProperty]
    private string _name;

    partial void OnNameChanged(string value) => _onRenamed?.Invoke();

    /// <summary>One end of the chord.</summary>
    public GeoPoint A { get; set; }

    /// <summary>The other end.</summary>
    public GeoPoint B { get; set; }

    /// <summary>The gate.</summary>
    public SectorGate ToGate() => new(Name, A, B);
}

/// <summary>What to do with unsaved track edits before they would be lost.</summary>
public enum UnsavedChoice
{
    /// <summary>Save them, then carry on.</summary>
    Save,

    /// <summary>Carry on and lose them.</summary>
    Discard,

    /// <summary>Stay where we are.</summary>
    Cancel,
}

/// <summary>
/// The Sectors tab: pick or create a track for the session, place sector
/// gates on the map, and see the split times of the selected laps.
/// </summary>
public sealed partial class SectorsViewModel : AnalysisTabViewModel<SectorsViewModel.Parameters, SplitTable>
{
    private const int MaxUndo = 50;

    private readonly SessionViewModel _a;
    private readonly TrackStore _store;
    private readonly Stack<List<SectorGate>> _undo = new();
    private bool _suppressTrackChange;
    private bool _suppressDirty;
    private TrackDefinition? _loadedTrack;

    /// <summary>Create the tab.</summary>
    public SectorsViewModel(SessionViewModel a, TrackStore store)
    {
        _a = a;
        _store = store;
        Tracks = new ObservableCollection<TrackDefinition>(_store.LoadAll());
        Commands = new[]
        {
            new TabCommand("_Save track", "Cmd+S", SaveTrackCommand),
            new TabCommand("_Undo gate edit", "Cmd+Z", UndoCommand),
        };
    }

    /// <inheritdoc />
    public override string Header => "Sectors";

    /// <inheritdoc />
    public override IReadOnlyList<TabCommand> Commands { get; }

    /// <summary>
    /// Asks what to do with unsaved edits to the named track (supplied by the
    /// view). Without it, as in tests, unsaved edits are discarded.
    /// </summary>
    public Func<string, Task<UnsavedChoice>>? ConfirmUnsaved { get; set; }

    /// <summary>Known tracks.</summary>
    public ObservableCollection<TrackDefinition> Tracks { get; }

    /// <summary>The track being used/edited, as stored (its edited name and gates are <see cref="TrackName"/> and <see cref="Gates"/>).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasTrack))]
    [NotifyCanExecuteChangedFor(nameof(SaveTrackCommand))]
    private TrackDefinition? _selectedTrack;

    /// <summary>Whether a track is selected.</summary>
    public bool HasTrack => SelectedTrack is not null;

    /// <summary>
    /// Editable name of the selected track. It is its own field: the stored
    /// track only takes it on save, so typing never touches the gates being
    /// placed.
    /// </summary>
    [ObservableProperty]
    private string _trackName = string.Empty;

    partial void OnTrackNameChanged(string value)
    {
        if (!_suppressDirty && SelectedTrack is not null)
        {
            IsDirty = true;
        }
    }

    /// <summary>Gates of the selected track, editable.</summary>
    public ObservableCollection<GateViewModel> Gates { get; } = new();

    /// <summary>The gate selected in the editor.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelectedGate))]
    [NotifyCanExecuteChangedFor(nameof(DeleteGateCommand))]
    private GateViewModel? _selectedGate;

    /// <summary>Whether a gate is selected.</summary>
    public bool HasSelectedGate => SelectedGate is not null;

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
    [NotifyCanExecuteChangedFor(nameof(SaveTrackCommand))]
    private bool _isDirty;

    partial void OnSelectedTrackChanged(TrackDefinition? oldValue, TrackDefinition? newValue)
    {
        if (_suppressTrackChange)
        {
            return;
        }

        if (IsDirty && oldValue is not null && ConfirmUnsaved is { } confirm)
        {
            _ = SwitchTrackAsync(oldValue, newValue, confirm);
            return;
        }

        LoadTrack(newValue);
    }

    /// <summary>The user picked another track while this one has unsaved edits: ask before they are lost.</summary>
    private async Task SwitchTrackAsync(TrackDefinition from, TrackDefinition? to, Func<string, Task<UnsavedChoice>> confirm)
    {
        // Until the answer is in, the edited track stays the one on screen.
        _suppressTrackChange = true;
        SelectedTrack = from;
        _suppressTrackChange = false;

        var choice = await confirm(TrackName);
        if (choice == UnsavedChoice.Cancel)
        {
            return;
        }

        if (choice == UnsavedChoice.Save)
        {
            SaveTrack();
            // Saving replaced the list entry; a switch to the track's own old record means "stay".
            if (to is not null && to.Id == from.Id)
            {
                return;
            }
        }

        _suppressTrackChange = true;
        SelectedTrack = to;
        _suppressTrackChange = false;
        LoadTrack(to);
    }

    /// <summary>Show <paramref name="track"/>'s stored name and gates, dropping any edits.</summary>
    private void LoadTrack(TrackDefinition? track)
    {
        _loadedTrack = track;
        _suppressDirty = true;
        TrackName = track?.Name ?? string.Empty;
        Gates.Clear();
        if (track is not null)
        {
            foreach (var g in track.Gates)
            {
                Gates.Add(NewGate(g));
            }
        }

        _suppressDirty = false;
        _undo.Clear();
        UndoCommand.NotifyCanExecuteChanged();
        SelectedGate = null;
        IsDirty = false;
        RebuildReferencePath();
        OnSelectionChanged();
    }

    private GateViewModel NewGate(SectorGate gate) => new(gate, OnGateRenamed);

    private void OnGateRenamed()
    {
        if (!_suppressDirty)
        {
            IsDirty = true;
            OnSelectionChanged();
        }
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
        _refLat = Array.Empty<double>();
        _refLon = Array.Empty<double>();
        if (_a.Log is null || GpsChannels() is not { } gps)
        {
            return;
        }

        var laps = _a.SelectedIn(Scope);
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

    /// <summary>
    /// The name a new track takes from a session: the venue of an AIM file,
    /// the track's display name of an iRacing one, else "New track".
    /// </summary>
    public static string TrackNameFor(IReadOnlyDictionary<string, string> metadata)
    {
        foreach (var key in new[] { "Venue", "track_display_name" })
        {
            if (metadata.TryGetValue(key, out var name) && name.Length > 0)
            {
                return name;
            }
        }

        return "New track";
    }

    /// <summary>Create a new track anchored at the session's first GPS fix and select it.</summary>
    [RelayCommand]
    public void NewTrack()
    {
        if (_a.Log is null || GpsChannels() is not { } gps || FirstFix(_a.Log, gps) is not { } anchor)
        {
            SetStatus("Load a session with GPS to create a track", StatusKind.Hint);
            return;
        }

        var name = TrackNameFor(_a.Log.Metadata);
        var id = TrackStore.IdFor(name);
        if (Tracks.Any(t => t.Id == id))
        {
            id += "_" + DateTime.Now.ToString("yyyyMMddHHmm", CultureInfo.InvariantCulture);
        }

        var track = new TrackDefinition(id, name, anchor, Array.Empty<SectorGate>());
        Tracks.Add(track);
        SelectedTrack = track;
        IsDirty = true;
        SetStatus($"New track '{name}' — click the line to add gates, then Save", StatusKind.Hint);
    }

    /// <summary>Add a gate across the reference path at sample <paramref name="index"/>.</summary>
    public void AddGateAt(int index)
    {
        if (SelectedTrack is null || _refLat.Length == 0)
        {
            return;
        }

        PushUndo();
        index = Math.Clamp(index, 0, _refLat.Length - 1);
        var gate = SectorTimes.GateAcrossPath(_refLat, _refLon, index, $"Gate {Gates.Count + 1}", SelectedTrack.Anchor);
        var vm = NewGate(gate);
        // Keep gates in lap order by the sample they sit at.
        var position = GateOrder().Count(i => i < index);
        Gates.Insert(position, vm);
        SelectedGate = vm;
        TouchTrack();
    }

    /// <summary>
    /// Call once before a drag starts moving a gate, so the whole drag is one
    /// undo step rather than one per pointer event.
    /// </summary>
    public void BeginGateMove() => PushUndo();

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
        if (!ordered.SequenceEqual(Gates))
        {
            Gates.Clear();
            foreach (var g in ordered)
            {
                Gates.Add(g);
            }

            SelectedGate = selected;
        }

        TouchTrack();
    }

    /// <summary>Move the selected gate <paramref name="samples"/> samples along the reference lap (arrow keys).</summary>
    public void NudgeSelectedGate(int samples)
    {
        if (SelectedGate is null)
        {
            return;
        }

        PushUndo();
        MoveSelectedGateTo(NearestIndex(SelectedGate) + samples);
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
    [RelayCommand(CanExecute = nameof(HasSelectedGate))]
    public void DeleteGate()
    {
        if (SelectedGate is null)
        {
            return;
        }

        PushUndo();
        Gates.Remove(SelectedGate);
        SelectedGate = null;
        TouchTrack();
    }

    private void PushUndo()
    {
        _undo.Push(Gates.Select(g => g.ToGate()).ToList());
        while (_undo.Count > MaxUndo)
        {
            // Stack has no trim; rebuild without the oldest entry.
            var kept = _undo.Take(MaxUndo).Reverse().ToList();
            _undo.Clear();
            foreach (var entry in kept)
            {
                _undo.Push(entry);
            }
        }

        UndoCommand.NotifyCanExecuteChanged();
    }

    /// <summary>Whether there is a gate edit to undo.</summary>
    public bool CanUndo => _undo.Count > 0;

    /// <summary>Take back the last gate edit (add, move or delete).</summary>
    [RelayCommand(CanExecute = nameof(CanUndo))]
    public void Undo()
    {
        if (_undo.Count == 0)
        {
            return;
        }

        var gates = _undo.Pop();
        _suppressDirty = true;
        Gates.Clear();
        foreach (var g in gates)
        {
            Gates.Add(NewGate(g));
        }

        _suppressDirty = false;
        SelectedGate = null;
        UndoCommand.NotifyCanExecuteChanged();
        TouchTrack();
    }

    private bool CanSaveTrack => SelectedTrack is not null && IsDirty;

    /// <summary>Store the track (user tracks file).</summary>
    [RelayCommand(CanExecute = nameof(CanSaveTrack))]
    public void SaveTrack()
    {
        if (SelectedTrack is null)
        {
            return;
        }

        var track = CurrentTrack();
        _store.Save(track);
        _suppressTrackChange = true;
        // Replace by id: the selected record may be a different instance from the list's.
        var index = Tracks.Select((t, i) => (t, i)).Where(x => x.t.Id == track.Id).Select(x => x.i).DefaultIfEmpty(-1).First();
        if (index >= 0)
        {
            Tracks[index] = track;
        }
        else
        {
            Tracks.Add(track);
        }

        SelectedTrack = track;
        _loadedTrack = track;
        _suppressTrackChange = false;
        IsDirty = false;
        SetStatus($"Track '{track.Name}' saved with {track.Gates.Count} gates", StatusKind.Done);
    }

    /// <summary>The selected track with the edited name and gates.</summary>
    public TrackDefinition CurrentTrack()
    {
        var name = TrackName.Trim();
        return SelectedTrack! with { Name = name.Length > 0 ? name : SelectedTrack!.Name, Gates = Gates.Select(g => g.ToGate()).ToList() };
    }

    private void TouchTrack()
    {
        IsDirty = true;
        OnSelectionChanged();
    }

    /// <summary>Everything one analysis reads, snapshotted on the UI thread.</summary>
    public sealed record Parameters(LogFile Log, List<int> Laps, TrackDefinition Track, string Lat, string Lon);

    /// <inheritdoc />
    protected override Parameters? GetParameters()
    {
        if (_a.Log is null)
        {
            return Hint(NoSessionHint);
        }

        if (GpsChannels() is not { } gps)
        {
            return Hint("No GPS channels in the session");
        }

        if (SelectedTrack is null)
        {
            return Hint("Select a track or create one for this session");
        }

        var laps = _a.SelectedIn(Scope);
        if (laps.Count == 0)
        {
            return Hint("Select laps to analyze");
        }

        return new Parameters(_a.Log, laps, CurrentTrack(), gps.Lat, gps.Lon);
    }

    /// <inheritdoc />
    protected override SplitTable Analyze(Parameters p, CancellationToken cancellation) =>
        SectorTimes.Build(p.Log, p.Laps, p.Track, p.Lat, p.Lon, cancellation);

    /// <inheritdoc />
    protected override void Display(Parameters p, SplitTable table)
    {
        Table = table;
        var incomplete = table.Laps.Count(l => !l.Complete);
        SetStatus(
            incomplete == 0
                ? $"{table.Laps.Count} laps, {table.Track.SectorCount} sectors — theoretical best {LapItemViewModel.FormatLapTime(table.TheoreticalBestS)}"
                : $"{incomplete} of {table.Laps.Count} laps miss a gate — move the gates onto the racing line",
            incomplete == 0 ? StatusKind.Done : StatusKind.Hint);
    }

    /// <inheritdoc />
    protected override void ClearDisplay() => Table = null;

    /// <summary>
    /// The split table as tab-separated text: a header row, one row per lap
    /// and the theoretical best, ready to paste into a spreadsheet.
    /// </summary>
    public string TableAsTsv()
    {
        if (Table is not { } t)
        {
            return string.Empty;
        }

        static string Time(double s) => double.IsNaN(s) ? string.Empty : s >= 60 ? LapItemViewModel.FormatLapTime(s) : s.ToString("0.000", CultureInfo.InvariantCulture);
        var sectors = t.Track.SectorCount;
        var lines = new List<string>
        {
            string.Join("\t", new[] { "Lap" }.Concat(Enumerable.Range(1, sectors).Select(i => $"S{i}")).Append("Lap time")),
        };
        foreach (var lap in t.Laps)
        {
            lines.Add(string.Join("\t", new[] { lap.LapNum.ToString(CultureInfo.InvariantCulture) }.Concat(lap.Sectors.Select(Time)).Append(Time(lap.LapTimeS))));
        }

        lines.Add(string.Join("\t", new[] { "Theoretical best" }.Concat(t.BestSector.Select(Time)).Append(Time(t.TheoreticalBestS))));
        return string.Join("\n", lines);
    }
}
