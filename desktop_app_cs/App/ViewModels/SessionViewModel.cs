using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>The two session slots.</summary>
public enum SessionSlot
{
    /// <summary>The primary session.</summary>
    A,

    /// <summary>The comparison session.</summary>
    B,
}

/// <summary>One lap row with its checkbox.</summary>
public sealed partial class LapItemViewModel : ObservableObject
{
    private readonly Action<LapItemViewModel> _onToggled;

    /// <summary>Create a row for <paramref name="lap"/>; <paramref name="gapS"/> is its time behind the session's best lap.</summary>
    public LapItemViewModel(Lap lap, double? gapS, Action<LapItemViewModel> onToggled)
    {
        Lap = lap;
        _onToggled = onToggled;
        IsBest = gapS == 0;
        GapText = gapS switch
        {
            null => string.Empty,
            0 => "best",
            _ => "+" + gapS.Value.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    /// <summary>The lap.</summary>
    public Lap Lap { get; }

    /// <summary>Lap number.</summary>
    public int Num => Lap.Num;

    /// <summary>"Lap 14".</summary>
    public string Name => $"Lap {Lap.Num}";

    /// <summary>"2:05.056".</summary>
    public string TimeText => FormatLapTime(Lap.LapTimeS);

    /// <summary>"best" for the fastest lap, "+0.412" for the others, empty for a lap without a time.</summary>
    public string GapText { get; }

    /// <summary>Whether this is the session's fastest lap.</summary>
    public bool IsBest { get; }

    /// <summary>
    /// How the lap's trace is drawn (colour and dash), shown as the key
    /// beside the row so the lap list is the legend; null when the lap is
    /// not selected or the active tab draws no per-lap traces.
    /// </summary>
    [ObservableProperty]
    private LapStyle? _style;

    private bool _isSelected;

    /// <summary>Checkbox state; toggling it raises the panel's selection callback.</summary>
    public bool IsSelected
    {
        get => _isSelected;
        set
        {
            if (SetProperty(ref _isSelected, value))
            {
                _onToggled(this);
            }
        }
    }

    /// <summary>Set the checkbox without raising the selection callback.</summary>
    public void SetSelectedSilently(bool value) => SetProperty(ref _isSelected, value, nameof(IsSelected));

    /// <summary>m:ss.sss, as the Python app shows it.</summary>
    public static string FormatLapTime(double seconds)
    {
        var minutes = (int)(seconds / 60);
        return $"{minutes}:{seconds - minutes * 60:00.000}";
    }
}

/// <summary>
/// A session slot (A or B): loads a file, resolves its profile, lists laps
/// with checkboxes and the Best / Top 103 % / All / None selectors; port of
/// the Python <c>SessionPanel</c>. A session keeps one lap selection per
/// <see cref="LapScope"/>: the laps overlaid as traces and the laps the
/// statistical tabs aggregate are different questions, so switching tabs
/// swaps which selection the checkboxes show instead of making the user
/// re-pick.
/// </summary>
public sealed partial class SessionViewModel : ObservableObject
{
    private readonly ProfileStore _profiles;
    private readonly LapStyleAllocator _styles;
    private readonly Dictionary<LapScope, HashSet<int>> _selected = new()
    {
        [LapScope.Analysis] = new HashSet<int>(),
        [LapScope.Traces] = new HashSet<int>(),
    };

    /// <summary>Create slot <paramref name="slot"/> titled <paramref name="title"/>.</summary>
    public SessionViewModel(string title, ProfileStore profiles, SessionSlot slot = SessionSlot.A, LapStyleAllocator? styles = null)
    {
        Title = title;
        Slot = slot;
        _profiles = profiles;
        _styles = styles ?? new LapStyleAllocator();
        _styles.Changed += ShowLapStyles; // the other session's selection can move this one's laps to other styles
    }

    /// <summary>Panel title, e.g. "Session A".</summary>
    public string Title { get; }

    /// <summary>Which slot this is.</summary>
    public SessionSlot Slot { get; }

    /// <summary>Raised after a file loads successfully and after the session closes.</summary>
    public event Action? SessionLoaded;

    /// <summary>Raised when the lap selection of a scope changes.</summary>
    public event Action<LapScope>? SelectionChanged;

    /// <summary>
    /// Opens the platform file picker and returns the chosen path (null:
    /// cancelled). Supplied by the view; null in tests and headless use.
    /// </summary>
    public Func<Task<string?>>? PickFile { get; set; }

    /// <summary>The loaded session, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSession))]
    [NotifyCanExecuteChangedFor(nameof(CloseCommand))]
    private LogFile? _log;

    /// <summary>Path of the loaded file.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(FileName))]
    private string? _filePath;

    /// <summary>Logger, vehicle and driver of the loaded file, e.g. "Logger 6701209 · Inferno 86 · A Driver".</summary>
    [ObservableProperty]
    private string _details = string.Empty;

    /// <summary>Why the last load failed, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    private string _error = string.Empty;

    /// <summary>True while a file is being parsed.</summary>
    [ObservableProperty]
    private bool _isLoading;

    /// <summary>The vehicle profile resolved for the session, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProfileName))]
    private VehicleProfile? _profile;

    /// <summary>Whether the active tab draws per-lap traces, so selected laps show their colour key.</summary>
    [ObservableProperty]
    private bool _showLapColors;

    partial void OnShowLapColorsChanged(bool value) => Restyle();

    /// <summary>Logger ID from the file metadata.</summary>
    public string? LoggerId { get; private set; }

    /// <summary>Vehicle name from the metadata or Race Studio file name.</summary>
    public string? VehicleName { get; private set; }

    /// <summary>Whether a session is loaded.</summary>
    public bool HasSession => Log is not null;

    /// <summary>Whether the last load failed.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>Name of the loaded file, or empty.</summary>
    public string FileName => FilePath is null ? string.Empty : Path.GetFileName(FilePath);

    /// <summary>"Profile: Toyota 86 ZN6", or a note that the defaults are in use.</summary>
    public string ProfileName => Profile is null ? "No vehicle profile (default channel names)" : $"Profile: {Profile.Name}";

    /// <summary>Lap rows.</summary>
    public ObservableCollection<LapItemViewModel> Laps { get; } = new();

    private LapScope _scope = LapScope.Analysis;

    /// <summary>
    /// Which selection the checkboxes show and edit. Switching it swaps the
    /// ticks; it does not raise <see cref="SelectionChanged"/>, since neither
    /// selection changed.
    /// </summary>
    public LapScope Scope
    {
        get => _scope;
        set
        {
            if (SetProperty(ref _scope, value))
            {
                ShowScope();
            }
        }
    }

    /// <summary>Selected lap numbers of the shown scope, in table order.</summary>
    public List<int> SelectedLaps => SelectedIn(Scope);

    /// <summary>Selected lap numbers of <paramref name="scope"/>, in table order.</summary>
    public List<int> SelectedIn(LapScope scope) => Laps.Where(l => _selected[scope].Contains(l.Num)).Select(l => l.Num).ToList();

    /// <summary>File name without extension, or "Session".</summary>
    public string SessionLabel => FilePath is null ? "Session" : Path.GetFileNameWithoutExtension(FilePath);

    /// <summary>The style of lap <paramref name="num"/>'s trace, or null when it is not selected in any scope.</summary>
    public LapStyle? StyleOf(int num) => _styles.Get(Slot.ToString(), num);

    /// <summary>Pick a file with the platform dialog and load it.</summary>
    [RelayCommand]
    public async Task BrowseAsync()
    {
        if (PickFile is { } pick && await pick() is { } path)
        {
            await LoadFileAsync(path);
        }
    }

    /// <summary>Load and post-process a telemetry file off the UI thread.</summary>
    [RelayCommand]
    public async Task LoadFileAsync(string path)
    {
        IsLoading = true;
        Error = string.Empty;
        try
        {
            var log = await Task.Run(() => SessionLoader.Load(path, _profiles));
            Apply(log, path);
        }
        catch (Exception e)
        {
            Clear();
            Error = $"Could not open {Path.GetFileName(path)}: {e.Message}";
            SessionLoaded?.Invoke();
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>Unload the session.</summary>
    [RelayCommand(CanExecute = nameof(HasSession))]
    public void Close()
    {
        Clear();
        Error = string.Empty;
        SessionLoaded?.Invoke();
    }

    private void Clear()
    {
        Log = null;
        FilePath = null;
        Profile = null;
        LoggerId = null;
        VehicleName = null;
        Details = string.Empty;
        Laps.Clear();
        foreach (var set in _selected.Values)
        {
            set.Clear();
        }

        _styles.Assign(Slot.ToString(), Array.Empty<int>(), Array.Empty<int>());
    }

    /// <summary>Use an already-loaded session (tests, demo mode).</summary>
    public void Apply(LogFile log, string path)
    {
        Error = string.Empty;
        Log = log;
        FilePath = path;
        LoggerId = LogIdentity.GetLoggerId(log.Metadata);
        VehicleName = LogIdentity.GetVehicleName(log.Metadata, Path.GetFileName(path));
        Profile = LoggerId is null ? null : _profiles.GetProfileForLogger(LoggerId, VehicleName);

        var parts = new List<string>();
        if (LoggerId is not null)
        {
            parts.Add($"Logger {LoggerId}");
        }

        // An iRacing file has no Race Studio vehicle name; libibt derives the car's from the session info.
        var vehicle = VehicleName ?? (log.Metadata.TryGetValue("car_name", out var car) && car.Length > 0 ? car : null);
        if (vehicle is not null)
        {
            parts.Add(vehicle);
        }

        foreach (var key in new[] { "Driver", "Driver Name", "driver_name" })
        {
            if (log.Metadata.TryGetValue(key, out var driver) && driver.Length > 0)
            {
                parts.Add(driver);
                break;
            }
        }

        Details = string.Join(" · ", parts);

        var timed = log.Laps.Where(l => l.LapTimeS > 0).ToList();
        var best = timed.Count > 0 ? timed.Min(l => l.LapTimeS) : (double?)null;
        Laps.Clear();
        foreach (var lap in log.Laps)
        {
            Laps.Add(new LapItemViewModel(lap, lap.LapTimeS > 0 && best is not null ? lap.LapTimeS - best.Value : null, OnLapToggled));
        }

        // Defaults: the statistical tabs get every representative lap, the trace overlay the best two.
        var byTime = timed.OrderBy(l => l.LapTimeS).Select(l => l.Num).ToList();
        _selected[LapScope.Analysis] = best is null ? new HashSet<int>() : timed.Where(l => l.LapTimeS <= best.Value * 1.03).Select(l => l.Num).ToHashSet();
        _selected[LapScope.Traces] = byTime.Take(2).ToHashSet();
        ShowScope();

        SessionLoaded?.Invoke();
        SelectionChanged?.Invoke(LapScope.Analysis);
        SelectionChanged?.Invoke(LapScope.Traces);
    }

    /// <summary>Tick the rows of the current scope and refresh the colour keys.</summary>
    private void ShowScope()
    {
        var selected = _selected[Scope];
        foreach (var lap in Laps)
        {
            lap.SetSelectedSilently(selected.Contains(lap.Num));
        }

        Restyle();
    }

    /// <summary>
    /// Re-allocate the lap styles (fastest first, so the best lap takes the
    /// first hue) and show them on the selected rows.
    /// </summary>
    private void Restyle()
    {
        var times = Laps.ToDictionary(l => l.Num, l => l.Lap.LapTimeS > 0 ? l.Lap.LapTimeS : double.MaxValue);
        double Rank(int num) => times.GetValueOrDefault(num, double.MaxValue);
        _styles.Assign(
            Slot.ToString(),
            _selected[LapScope.Traces].OrderBy(Rank).ToList(),
            _selected[LapScope.Analysis].OrderBy(Rank).ToList());
    }

    /// <summary>Show each selected row's current style as its colour key.</summary>
    private void ShowLapStyles()
    {
        foreach (var lap in Laps)
        {
            lap.Style = ShowLapColors && lap.IsSelected ? _styles.Get(Slot.ToString(), lap.Num) : null;
        }
    }

    private void OnLapToggled(LapItemViewModel lap)
    {
        if (lap.IsSelected)
        {
            _selected[Scope].Add(lap.Num);
        }
        else
        {
            _selected[Scope].Remove(lap.Num);
        }

        Restyle();
        SelectionChanged?.Invoke(Scope);
    }

    private void SetAll(Func<LapItemViewModel, bool> selected)
    {
        _selected[Scope] = Laps.Where(selected).Select(l => l.Num).ToHashSet();
        ShowScope();
        SelectionChanged?.Invoke(Scope);
    }

    /// <summary>Select every lap.</summary>
    [RelayCommand]
    public void SelectAll() => SetAll(_ => true);

    /// <summary>Deselect every lap.</summary>
    [RelayCommand]
    public void SelectNone() => SetAll(_ => false);

    /// <summary>Select only the fastest lap.</summary>
    [RelayCommand]
    public void SelectBest()
    {
        var best = BestLapNum();
        if (best is not null)
        {
            SetAll(l => l.Num == best);
        }
    }

    /// <summary>Select the laps within 103 % of the fastest.</summary>
    [RelayCommand]
    public void SelectTop103()
    {
        var best = BestLapNum();
        if (best is null)
        {
            return;
        }

        var threshold = Laps.First(l => l.Num == best).Lap.LapTimeS * 1.03;
        SetAll(l => l.Lap.LapTimeS > 0 && l.Lap.LapTimeS <= threshold);
    }

    private int? BestLapNum() =>
        Laps.Where(l => l.Lap.LapTimeS > 0).OrderBy(l => l.Lap.LapTimeS).Select(l => (int?)l.Num).FirstOrDefault();
}
