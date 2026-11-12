using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>One lap row with its checkbox.</summary>
public sealed partial class LapItemViewModel : ObservableObject
{
    private readonly Action _onToggled;

    /// <summary>Create a row for <paramref name="lap"/>.</summary>
    public LapItemViewModel(Lap lap, Action onToggled)
    {
        Lap = lap;
        _onToggled = onToggled;
    }

    /// <summary>The lap.</summary>
    public Lap Lap { get; }

    /// <summary>Lap number.</summary>
    public int Num => Lap.Num;

    /// <summary>"Lap 14: 2:05.056".</summary>
    public string Label => $"Lap {Lap.Num}: {FormatLapTime(Lap.LapTimeS)}";

    private string? _traceColor;

    /// <summary>
    /// Hex colour of this lap's trace on the Time/Distance chart, set after
    /// each analysis so the lap list doubles as the legend; null when the lap
    /// is not drawn.
    /// </summary>
    public string? TraceColor
    {
        get => _traceColor;
        set
        {
            if (SetProperty(ref _traceColor, value))
            {
                OnPropertyChanged(nameof(TraceBrush));
            }
        }
    }

    /// <summary>Swatch brush for the lap list (transparent when not drawn).</summary>
    public Avalonia.Media.IBrush TraceBrush =>
        _traceColor is null ? Avalonia.Media.Brushes.Transparent : new Avalonia.Media.SolidColorBrush(Avalonia.Media.Color.Parse(_traceColor));

    private bool _isSelected;

    /// <summary>Checkbox state; toggling it raises the panel's selection callback.</summary>
    public bool IsSelected
    {
        get => _isSelected;
        set
        {
            if (SetProperty(ref _isSelected, value))
            {
                _onToggled();
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
/// the Python <c>SessionPanel</c>.
/// </summary>
public sealed partial class SessionViewModel : ObservableObject
{
    private readonly ProfileStore _profiles;
    private bool _suppressSelection;

    /// <summary>Create a slot titled <paramref name="title"/>.</summary>
    public SessionViewModel(string title, ProfileStore profiles)
    {
        Title = title;
        _profiles = profiles;
    }

    /// <summary>Panel title, e.g. "SESSION A (Primary)".</summary>
    public string Title { get; }

    /// <summary>Raised after a file loads successfully.</summary>
    public event Action? SessionLoaded;

    /// <summary>Raised when the lap selection changes.</summary>
    public event Action? SelectionChanged;

    /// <summary>The loaded session, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSession))]
    private LogFile? _log;

    /// <summary>Path of the loaded file.</summary>
    [ObservableProperty]
    private string? _filePath;

    /// <summary>"File: … | Logger: … | Vehicle: … | Driver: …", or an error.</summary>
    [ObservableProperty]
    private string _fileLabel = "No file loaded";

    /// <summary>True while a file is being parsed.</summary>
    [ObservableProperty]
    private bool _isLoading;

    /// <summary>The vehicle profile resolved for the session, or null.</summary>
    [ObservableProperty]
    private VehicleProfile? _profile;

    /// <summary>Logger ID from the file metadata.</summary>
    public string? LoggerId { get; private set; }

    /// <summary>Vehicle name from the metadata or Race Studio file name.</summary>
    public string? VehicleName { get; private set; }

    /// <summary>Whether a session is loaded.</summary>
    public bool HasSession => Log is not null;

    /// <summary>Lap rows.</summary>
    public ObservableCollection<LapItemViewModel> Laps { get; } = new();

    /// <summary>Selected lap numbers in table order.</summary>
    public List<int> SelectedLaps => Laps.Where(l => l.IsSelected).Select(l => l.Num).ToList();

    /// <summary>File name without extension, or "Session".</summary>
    public string SessionLabel => FilePath is null ? "Session" : Path.GetFileNameWithoutExtension(FilePath);

    /// <summary>Load and post-process a telemetry file off the UI thread.</summary>
    [RelayCommand]
    public async Task LoadFileAsync(string path)
    {
        IsLoading = true;
        FileLabel = $"Loading {Path.GetFileName(path)}…";
        try
        {
            var log = await Task.Run(() => SessionLoader.Load(path, _profiles));
            Apply(log, path);
        }
        catch (Exception e)
        {
            Log = null;
            FilePath = null;
            Profile = null;
            LoggerId = null;
            VehicleName = null;
            Laps.Clear();
            FileLabel = $"Error: {e.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>Use an already-loaded session (tests, demo mode).</summary>
    public void Apply(LogFile log, string path)
    {
        Log = log;
        FilePath = path;
        LoggerId = LogIdentity.GetLoggerId(log.Metadata);
        VehicleName = LogIdentity.GetVehicleName(log.Metadata, Path.GetFileName(path));
        Profile = LoggerId is null ? null : _profiles.GetProfileForLogger(LoggerId, VehicleName);

        var parts = new List<string> { $"File: {Path.GetFileName(path)}" };
        if (LoggerId is not null)
        {
            parts.Add($"Logger: {LoggerId}");
        }

        // An iRacing file has no Race Studio vehicle name; libibt derives the car's from the session info.
        var vehicle = VehicleName ?? (log.Metadata.TryGetValue("car_name", out var car) && car.Length > 0 ? car : null);
        if (vehicle is not null)
        {
            parts.Add($"Vehicle: {vehicle}");
        }

        foreach (var key in new[] { "Driver", "Driver Name", "driver_name" })
        {
            if (log.Metadata.TryGetValue(key, out var driver) && driver.Length > 0)
            {
                parts.Add($"Driver: {driver}");
                break;
            }
        }

        FileLabel = string.Join(" | ", parts);

        _suppressSelection = true;
        Laps.Clear();
        foreach (var lap in log.Laps)
        {
            Laps.Add(new LapItemViewModel(lap, OnLapToggled));
        }

        _suppressSelection = false;
        SessionLoaded?.Invoke();
        SelectTop103();
    }

    private void OnLapToggled()
    {
        if (!_suppressSelection)
        {
            SelectionChanged?.Invoke();
        }
    }

    private void SetAll(Func<LapItemViewModel, bool> selected)
    {
        foreach (var lap in Laps)
        {
            lap.SetSelectedSilently(selected(lap));
        }

        SelectionChanged?.Invoke();
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
