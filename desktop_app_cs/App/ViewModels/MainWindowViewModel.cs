using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>
/// The window: two session slots, the analysis tabs, and the routing of
/// session/selection events to the tabs (port of <c>InfernoAnalyzerApp</c>).
/// </summary>
public sealed partial class MainWindowViewModel : ObservableObject
{
    /// <summary>Create the window model over <paramref name="profiles"/>.</summary>
    public MainWindowViewModel(ProfileStore? profiles = null)
    {
        Profiles = profiles ?? new ProfileStore();
        SessionA = new SessionViewModel("SESSION A (Primary)", Profiles);
        SessionB = new SessionViewModel("SESSION B (Compare)", Profiles);
        TimeDistance = new TimeDistanceViewModel(SessionA, SessionB);
        Suspension = new SuspensionViewModel(SessionA, SessionB);
        RollGradient = new RollGradientViewModel(SessionA, SessionB, Suspension.SuspensionConfig);
        Driver = new DriverViewModel(SessionA, SessionB);
        TireGrip = new TireGripViewModel(SessionA, SessionB);
        Tabs = new ObservableCollection<AnalysisTabViewModel> { TimeDistance, Suspension, Driver, TireGrip, RollGradient };
        _activeTab = TimeDistance;
        TimeDistance.SetActive(true);

        SessionA.SessionLoaded += OnSessionLoaded;
        SessionB.SessionLoaded += OnSessionLoaded;
        SessionA.SelectionChanged += OnSelectionChanged;
        SessionB.SelectionChanged += OnSelectionChanged;
    }

    /// <summary>Profile store shared by the sessions and tabs.</summary>
    public ProfileStore Profiles { get; }

    /// <summary>Primary session.</summary>
    public SessionViewModel SessionA { get; }

    /// <summary>Comparison session.</summary>
    public SessionViewModel SessionB { get; }

    /// <summary>The Time/Distance tab.</summary>
    public TimeDistanceViewModel TimeDistance { get; }

    /// <summary>The Suspension tab.</summary>
    public SuspensionViewModel Suspension { get; }

    /// <summary>The Roll Gradient tab.</summary>
    public RollGradientViewModel RollGradient { get; }

    /// <summary>The Driver Consistency tab.</summary>
    public DriverViewModel Driver { get; }

    /// <summary>The Tire Grip tab.</summary>
    public TireGripViewModel TireGrip { get; }

    /// <summary>Analysis tabs in display order.</summary>
    public ObservableCollection<AnalysisTabViewModel> Tabs { get; }

    /// <summary>The visible tab.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Status))]
    private AnalysisTabViewModel _activeTab;

    /// <summary>Top panel collapsed to a one-line summary.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CollapsedSummary))]
    private bool _isTopCollapsed;

    /// <summary>The active tab's status line.</summary>
    public string Status => ActiveTab.Status;

    /// <summary>"A: file (n/m laps) | B: No file".</summary>
    public string CollapsedSummary
    {
        get
        {
            static string One(string slot, SessionViewModel s) =>
                s.Log is null ? $"{slot}: No file" : $"{slot}: {Path.GetFileName(s.FilePath)} ({s.SelectedLaps.Count}/{s.Laps.Count} laps)";
            return $"{One("A", SessionA)}  |  {One("B", SessionB)}";
        }
    }

    partial void OnActiveTabChanged(AnalysisTabViewModel? oldValue, AnalysisTabViewModel newValue)
    {
        if (oldValue is not null)
        {
            oldValue.PropertyChanged -= OnActiveTabPropertyChanged;
            oldValue.SetActive(false);
        }

        newValue.PropertyChanged += OnActiveTabPropertyChanged;
        newValue.SetActive(true);
        OnPropertyChanged(nameof(Status));
    }

    private void OnActiveTabPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(AnalysisTabViewModel.Status))
        {
            OnPropertyChanged(nameof(Status));
        }
    }

    /// <summary>Toggle the top panel.</summary>
    [RelayCommand]
    public void ToggleTopPanel() => IsTopCollapsed = !IsTopCollapsed;

    /// <summary>
    /// Save a vehicle profile for the session the active config panel is
    /// editing (A, or B when selected): every tab's channel names merged,
    /// the Suspension tab's motion ratios and the Roll Gradient tab's track
    /// widths, under the logger ID and vehicle name, as the Python app does.
    /// </summary>
    [RelayCommand]
    public void SaveProfile()
    {
        var session = ActiveTab.Config is ChannelConfigViewModel { ActiveSession: "B" } && SessionB.Log is not null ? "B" : "A";
        var slot = session == "B" ? SessionB : SessionA;
        if (slot.LoggerId is null)
        {
            ActiveTab.Status = "No logger ID — load a session first";
            return;
        }

        // Tabs share some keys (the shock pots appear on Suspension and Roll Gradient); the
        // first tab to define a key owns it, so the Suspension entries win for the shocks.
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var config in Tabs.Select(t => t.Config).OfType<ChannelConfigViewModel>())
        {
            foreach (var (key, value) in config.GetChannelNames(session))
            {
                names.TryAdd(key, value);
            }
        }

        var ratios = Suspension.SuspensionConfig.GetMotionRatios(session);
        var existing = slot.Profile;
        // Keep the loaded calibration only while the entered ratios still derive from it.
        var calibration = existing?.ShockCalibration is { } cal && cal.PotWheelRatios() == ratios ? cal : null;
        var profile = new VehicleProfile(slot.SessionLabel, names, ratios, calibration, RollGradient.RollConfig.GetTrackWidths(session));
        Profiles.SaveProfileForLogger(slot.LoggerId, profile, slot.VehicleName);
        var ident = $"logger {slot.LoggerId}" + (slot.VehicleName is null ? string.Empty : $" / {slot.VehicleName}");
        ActiveTab.Status = $"Profile saved for {ident}";
    }

    private void OnSessionLoaded()
    {
        foreach (var tab in Tabs)
        {
            tab.OnSessionLoaded();
        }

        OnPropertyChanged(nameof(CollapsedSummary));
    }

    private void OnSelectionChanged()
    {
        foreach (var tab in Tabs)
        {
            tab.OnSelectionChanged();
        }

        OnPropertyChanged(nameof(CollapsedSummary));
    }
}
