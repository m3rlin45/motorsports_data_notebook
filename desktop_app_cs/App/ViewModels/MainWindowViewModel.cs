using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>A tab's channel configuration as a page of the vehicle profile window.</summary>
/// <param name="Header">The tab's header.</param>
/// <param name="Config">The tab's config view model.</param>
public sealed record ConfigSection(string Header, ChannelConfigViewModel Config);

/// <summary>
/// The window: two session slots, the analysis tabs, and the routing of
/// session/selection events to the tabs (port of <c>InfernoAnalyzerApp</c>),
/// plus what the shell around them needs: opening files, recent files, the
/// vehicle profile editor and the settings that persist between runs.
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
        Suspension = new SuspensionViewModel(SessionA, SessionB);
        RollGradient = new RollGradientViewModel(SessionA, SessionB, Suspension.SuspensionConfig);
        Driver = new DriverViewModel(SessionA, SessionB);
        Tabs = new ObservableCollection<AnalysisTabViewModel> { TimeDistance, Suspension, Driver, RollGradient };
        ConfigSections = Tabs.Where(t => t.Config is ChannelConfigViewModel).Select(t => new ConfigSection(t.Header, (ChannelConfigViewModel)t.Config!)).ToList();
        _selectedConfigSection = ConfigSections.FirstOrDefault();

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

    /// <summary>The Suspension tab.</summary>
    public SuspensionViewModel Suspension { get; }

    /// <summary>The Roll Gradient tab.</summary>
    public RollGradientViewModel RollGradient { get; }

    /// <summary>The Driver Consistency tab.</summary>
    public DriverViewModel Driver { get; }

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
        if (newTab.Config is ChannelConfigViewModel config)
        {
            SelectedConfigSection = ConfigSections.FirstOrDefault(s => ReferenceEquals(s.Config, config)) ?? SelectedConfigSection;
        }
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

    /// <summary>The tabs' channel configurations, as the pages of the vehicle profile window.</summary>
    public IReadOnlyList<ConfigSection> ConfigSections { get; }

    /// <summary>The page shown in the vehicle profile window; follows the active tab.</summary>
    [ObservableProperty]
    private ConfigSection? _selectedConfigSection;

    /// <summary>The session the vehicle profile window edits; every page follows it.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsProfileSessionA))]
    [NotifyPropertyChangedFor(nameof(IsProfileSessionB))]
    [NotifyPropertyChangedFor(nameof(ProfileTarget))]
    private SessionSlot _profileSession = SessionSlot.A;

    partial void OnProfileSessionChanged(SessionSlot value)
    {
        foreach (var section in ConfigSections)
        {
            section.Config.ActiveSession = value;
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsProfileSessionA
    {
        get => ProfileSession == SessionSlot.A;
        set
        {
            if (value)
            {
                ProfileSession = SessionSlot.A;
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsProfileSessionB
    {
        get => ProfileSession == SessionSlot.B;
        set
        {
            if (value)
            {
                ProfileSession = SessionSlot.B;
            }
        }
    }

    /// <summary>Keep every page's Session B entries equal to Session A's.</summary>
    public bool SyncProfileWithA
    {
        get => ConfigSections.All(s => s.Config.SyncWithA);
        set
        {
            foreach (var section in ConfigSections)
            {
                section.Config.SyncWithA = value;
            }

            OnPropertyChanged();
        }
    }

    /// <summary>Which vehicle the profile window is editing, e.g. "Logger 6701209 · Inferno 86 (Profile: Toyota 86 ZN6)".</summary>
    public string ProfileTarget
    {
        get
        {
            var slot = ProfileSession == SessionSlot.B && SessionB.HasSession ? SessionB : SessionA;
            return slot.HasSession ? $"{slot.Details}  ({slot.ProfileName})" : "No session loaded";
        }
    }

    /// <summary>Raised when the vehicle profile window should be shown.</summary>
    public event Action? ProfileEditorRequested;

    /// <summary>Open the vehicle profile window on the active tab's page.</summary>
    [RelayCommand]
    public void EditProfile() => ProfileEditorRequested?.Invoke();

    /// <summary>Restore the package defaults on the shown page.</summary>
    [RelayCommand]
    public void ResetConfigSection() => SelectedConfigSection?.Config.Reset();

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

        if (!SessionB.HasSession && ProfileSession == SessionSlot.B)
        {
            ProfileSession = SessionSlot.A;
        }

        foreach (var tab in Tabs)
        {
            tab.OnSessionLoaded();
        }

        SwapSessionsCommand.NotifyCanExecuteChanged();
        OnPropertyChanged(nameof(WindowTitle));
        OnPropertyChanged(nameof(ProfileTarget));
        OnPropertyChanged(nameof(SyncProfileWithA));
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
