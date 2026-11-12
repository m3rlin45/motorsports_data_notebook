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
        TimeDistance = new TimeDistanceViewModel(SessionA);
        Tabs = new ObservableCollection<AnalysisTabViewModel> { TimeDistance };
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
