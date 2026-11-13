using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>
/// The window: two session slots, the analysis tabs, and the routing of
/// session/selection events to the tabs (port of <c>InfernoAnalyzerApp</c>),
/// plus what the shell around them needs: opening files, recent files and
/// the settings that persist between runs.
/// </summary>
public sealed partial class MainWindowViewModel : ObservableObject
{
    /// <summary>File extensions the app opens.</summary>
    public static readonly string[] TelemetryExtensions = { ".xrk", ".xrz", ".ibt" };

    private readonly SettingsStore _settingsStore;

    /// <summary>
    /// Create the window model over <paramref name="profiles"/>. Without
    /// <paramref name="settings"/> nothing is read from or written to disk
    /// (tests).
    /// </summary>
    public MainWindowViewModel(ProfileStore? profiles = null, SettingsStore? settings = null)
    {
        Profiles = profiles ?? new ProfileStore();
        _settingsStore = settings ?? new SettingsStore();
        var styles = new LapStyleAllocator();
        SessionA = new SessionViewModel("Session A", Profiles, SessionSlot.A, styles);
        SessionB = new SessionViewModel("Session B", Profiles, SessionSlot.B, styles);
        TimeDistance = new TimeDistanceViewModel(SessionA, SessionB, Settings);
        Tabs = new ObservableCollection<AnalysisTabViewModel> { TimeDistance };

        foreach (var tab in Tabs)
        {
            if (Settings.Tabs.TryGetValue(tab.Header, out var state))
            {
                tab.RestoreState(state);
            }
        }

        _isSidebarCollapsed = Settings.SidebarCollapsed;
        _theme = Settings.Theme;
        RecentFiles = new ObservableCollection<string>(Settings.RecentFiles);

        foreach (var session in new[] { SessionA, SessionB })
        {
            session.SessionLoaded += () => OnSessionLoaded(session);
            session.SelectionChanged += OnSelectionChanged;
            session.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName is nameof(SessionViewModel.IsLoading))
                {
                    OnPropertyChanged(nameof(IsBusy));
                }
            };
        }

        _activeTab = Tabs.FirstOrDefault(t => t.Header == Settings.LastTab) ?? TimeDistance;
        Activate(null, _activeTab);
    }

    /// <summary>Profile store shared by the sessions and tabs.</summary>
    public ProfileStore Profiles { get; }

    /// <summary>What is remembered between runs.</summary>
    public AppSettings Settings => _settingsStore.Settings;

    /// <summary>Primary session.</summary>
    public SessionViewModel SessionA { get; }

    /// <summary>Comparison session.</summary>
    public SessionViewModel SessionB { get; }

    /// <summary>The Time/Distance tab.</summary>
    public TimeDistanceViewModel TimeDistance { get; }

    /// <summary>Analysis tabs in display order.</summary>
    public ObservableCollection<AnalysisTabViewModel> Tabs { get; }

    /// <summary>Recently opened session files, newest first.</summary>
    public ObservableCollection<string> RecentFiles { get; }

    /// <summary>The visible tab.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsBusy))]
    private AnalysisTabViewModel _activeTab;

    /// <summary>Whether the session sidebar is hidden, giving the charts the full width.</summary>
    [ObservableProperty]
    private bool _isSidebarCollapsed;

    /// <summary>"System", "Light" or "Dark".</summary>
    [ObservableProperty]
    private string _theme;

    /// <summary>True while a file is being read or the visible tab is computing.</summary>
    public bool IsBusy => SessionA.IsLoading || SessionB.IsLoading || ActiveTab.IsAnalyzing;

    /// <summary>"Inferno Analyzer", followed by the loaded file names.</summary>
    public string WindowTitle
    {
        get
        {
            var files = new[] { SessionA.FileName, SessionB.FileName }.Where(f => f.Length > 0).ToList();
            return files.Count == 0 ? "Inferno Analyzer" : $"{string.Join(" vs ", files)} — Inferno Analyzer";
        }
    }

    partial void OnActiveTabChanged(AnalysisTabViewModel? oldValue, AnalysisTabViewModel newValue) => Activate(oldValue, newValue);

    private void Activate(AnalysisTabViewModel? oldTab, AnalysisTabViewModel newTab)
    {
        if (oldTab is not null)
        {
            oldTab.PropertyChanged -= OnActiveTabPropertyChanged;
            oldTab.SetActive(false);
        }

        newTab.PropertyChanged += OnActiveTabPropertyChanged;
        ShowTabLapSelection(newTab);
        newTab.SetActive(true);
    }

    /// <summary>The lap lists show the selection the tab works on, with colour keys when it draws one trace per lap.</summary>
    private void ShowTabLapSelection(AnalysisTabViewModel tab)
    {
        foreach (var session in new[] { SessionA, SessionB })
        {
            session.Scope = tab.Scope;
            session.ShowLapColors = tab.UsesLapColors;
        }
    }

    private void OnActiveTabPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(AnalysisTabViewModel.IsAnalyzing):
                OnPropertyChanged(nameof(IsBusy));
                break;
            case nameof(AnalysisTabViewModel.UsesLapColors):
                ShowTabLapSelection(ActiveTab);
                break;
        }
    }

    /// <summary>Show or hide the session sidebar.</summary>
    [RelayCommand]
    public void ToggleSidebar() => IsSidebarCollapsed = !IsSidebarCollapsed;

    /// <summary>Open a file into Session A with the platform dialog.</summary>
    [RelayCommand]
    public Task OpenSessionA() => SessionA.BrowseAsync();

    /// <summary>Open a file into Session B with the platform dialog.</summary>
    [RelayCommand]
    public Task OpenSessionB() => SessionB.BrowseAsync();

    /// <summary>Whether <paramref name="path"/> has one of the <see cref="TelemetryExtensions"/>.</summary>
    public static bool IsTelemetryFile(string? path) =>
        path is not null && TelemetryExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>
    /// Open dropped or command-line files: one goes to <paramref name="target"/>
    /// (Session A unless the drop landed on Session B's panel), two fill A
    /// and B. Files beyond the second and files of other types are ignored.
    /// </summary>
    public async Task OpenFilesAsync(IEnumerable<string> paths, SessionSlot target = SessionSlot.A)
    {
        var files = paths.Where(IsTelemetryFile).Take(2).ToList();
        if (files.Count == 2)
        {
            await SessionA.LoadFileAsync(files[0]);
            await SessionB.LoadFileAsync(files[1]);
        }
        else if (files.Count == 1)
        {
            await (target == SessionSlot.B ? SessionB : SessionA).LoadFileAsync(files[0]);
        }
    }

    /// <summary>Open a recent file into Session A.</summary>
    [RelayCommand]
    public Task OpenRecent(string path) => SessionA.LoadFileAsync(path);

    private bool BothLoaded => SessionA.HasSession && SessionB.HasSession;

    /// <summary>Exchange the two sessions (the comparison becomes the primary).</summary>
    [RelayCommand(CanExecute = nameof(BothLoaded))]
    public void SwapSessions()
    {
        if (SessionA.Log is not { } logA || SessionB.Log is not { } logB || SessionA.FilePath is not { } pathA || SessionB.FilePath is not { } pathB)
        {
            return;
        }

        SessionA.Apply(logB, pathB);
        SessionB.Apply(logA, pathA);
    }

    private void OnSessionLoaded(SessionViewModel session)
    {
        if (session.FilePath is { } path && File.Exists(path))
        {
            Settings.AddRecent(path);
            RecentFiles.Clear();
            foreach (var recent in Settings.RecentFiles)
            {
                RecentFiles.Add(recent);
            }
        }

        foreach (var tab in Tabs)
        {
            tab.OnSessionLoaded();
        }

        SwapSessionsCommand.NotifyCanExecuteChanged();
        OnPropertyChanged(nameof(WindowTitle));
    }

    private void OnSelectionChanged(LapScope scope)
    {
        foreach (var tab in Tabs.Where(t => t.Scope == scope))
        {
            tab.OnSelectionChanged();
        }
    }

    /// <summary>Write what should survive a restart into <see cref="Settings"/> and save them.</summary>
    public void SaveSettings()
    {
        Settings.SidebarCollapsed = IsSidebarCollapsed;
        Settings.Theme = Theme;
        Settings.LastTab = ActiveTab.Header;
        foreach (var tab in Tabs)
        {
            var state = new Dictionary<string, string>(StringComparer.Ordinal);
            tab.SaveState(state);
            Settings.Tabs[tab.Header] = state;
        }

        _settingsStore.Save();
    }
}
