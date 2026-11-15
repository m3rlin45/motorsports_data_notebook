using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.App.Series;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>A channel the strip chart can show, with its checkbox.</summary>
public sealed partial class ChannelChoiceViewModel : ObservableObject
{
    private readonly Action<ChannelChoiceViewModel> _onToggled;

    /// <summary>Create a choice for <paramref name="name"/>.</summary>
    public ChannelChoiceViewModel(string name, bool isSelected, Action<ChannelChoiceViewModel> onToggled)
    {
        Name = name;
        _isSelected = isSelected;
        _onToggled = onToggled;
    }

    /// <summary>Channel name.</summary>
    public string Name { get; }

    [ObservableProperty]
    private bool _isSelected;

    partial void OnIsSelectedChanged(bool value) => _onToggled(this);
}

/// <summary>
/// The Time/Distance tab: the selected laps of Session A overlaid as a strip
/// chart (one panel per chosen channel plus delta time), with the linked
/// crosshair. Also hosts the demo fixtures (`cs-run path.json`).
/// </summary>
public sealed partial class TimeDistanceViewModel : AnalysisTabViewModel<TimeDistanceViewModel.Parameters, Fixture>
{
    private readonly SessionViewModel _sessionA;
    private readonly AppSettings? _settings;
    private Fixture? _fixture;
    private string? _viewKey;
    private bool _suppressChannelEvents;

    /// <summary>
    /// Create the tab over <paramref name="sessionA"/>. With
    /// <paramref name="settings"/> the ticked channels are remembered per
    /// vehicle.
    /// </summary>
    public TimeDistanceViewModel(SessionViewModel sessionA, AppSettings? settings = null)
    {
        _sessionA = sessionA;
        _settings = settings;
        Commands = new[]
        {
            new TabCommand("Distance / time _axis", "F9", ToggleXAxisCommand),
        };
    }

    /// <inheritdoc />
    public override string Header => "Time / Distance";

    /// <inheritdoc />
    public override LapScope Scope => LapScope.Traces;

    /// <inheritdoc />
    public override bool UsesLapColors => true;

    /// <inheritdoc />
    public override IReadOnlyList<TabCommand> Commands { get; }

    /// <summary>The linked crosshair group the chart joins.</summary>
    public CursorLinkGroup Cursors { get; } = new();

    /// <summary>Channels available in Session A, alphabetically, with the plotted ones checked.</summary>
    public ObservableCollection<ChannelChoiceViewModel> Channels { get; } = new();

    /// <summary>The plotted channels in panel order (top to bottom); reorder with <see cref="MoveShown"/>.</summary>
    public ObservableCollection<ChannelChoiceViewModel> ShownChannels { get; } = new();

    /// <summary>Whether a session's channels are listed.</summary>
    public bool HasChannels => Channels.Count > 0;

    /// <summary>Whether any channel is plotted.</summary>
    public bool HasShownChannels => ShownChannels.Count > 0;

    /// <summary>Channel whose timecodes every series is resampled onto.</summary>
    [ObservableProperty]
    private string _timebase = ChannelKeys.DefaultChannelNames["gps_speed"];

    /// <summary>The chart to draw; the view renders it when it changes.</summary>
    [ObservableProperty]
    private StripChartSpec? _spec;

    /// <summary>"N laps" (diagnostics for the demo fixture and tests; not shown in the view).</summary>
    [ObservableProperty]
    private string _summary = "Load a session and select laps";

    /// <summary>X axis mode.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDistance))]
    [NotifyPropertyChangedFor(nameof(IsTime))]
    private XAxisMode _xAxis = XAxisMode.Distance;

    /// <summary>Radio binding for distance mode.</summary>
    public bool IsDistance
    {
        get => XAxis == XAxisMode.Distance;
        set
        {
            if (value)
            {
                XAxis = XAxisMode.Distance;
            }
        }
    }

    /// <summary>Radio binding for time mode.</summary>
    public bool IsTime
    {
        get => XAxis == XAxisMode.Time;
        set
        {
            if (value)
            {
                XAxis = XAxisMode.Time;
            }
        }
    }

    /// <summary>Switch between distance and time on the x-axis.</summary>
    [RelayCommand]
    public void ToggleXAxis() => XAxis = XAxis == XAxisMode.Distance ? XAxisMode.Time : XAxisMode.Distance;

    partial void OnXAxisChanged(XAxisMode value)
    {
        Cursors.SetCursor(null);
        if (_fixture is not null)
        {
            Show(_fixture);
        }
    }

    /// <summary>Show a dumped fixture instead of a session (demo mode).</summary>
    public void ShowFixture(Fixture fixture)
    {
        _viewKey = fixture.Source;
        EmptyMessage = null;
        Show(fixture);
        SetStatus("Fixture loaded", StatusKind.Done);
    }

    /// <summary>Settings key of a session's vehicle: the logger and vehicle that pick its profile.</summary>
    private static string? VehicleKey(SessionViewModel session) =>
        session.LoggerId is null ? null : $"{session.LoggerId}|{session.VehicleName}";

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        var log = _sessionA.Log;
        var profile = _sessionA.Profile;
        var names = profile?.ChannelNames ?? ChannelKeys.DefaultChannelNames;
        Timebase = names["gps_speed"];

        // What to plot, most specific first: what this vehicle had ticked last time, what is ticked now
        // (the next run of the same car), then speed, throttle and brake.
        var available = log?.ChannelNames.ToHashSet(StringComparer.Ordinal) ?? new HashSet<string>();
        var remembered = VehicleKey(_sessionA) is { } key && _settings is not null && _settings.ChannelsByVehicle.TryGetValue(key, out var saved) ? saved : null;
        var wanted = (remembered ?? ShownChannels.Select(c => c.Name)).Where(available.Contains).ToList();
        if (wanted.Count == 0)
        {
            wanted = new[] { "speed_kmh", names["throttle"], names["brake"] }.Where(available.Contains).ToList();
        }

        _suppressChannelEvents = true;
        Channels.Clear();
        ShownChannels.Clear();
        if (log is not null)
        {
            foreach (var name in log.ChannelNames.OrderBy(n => n, StringComparer.OrdinalIgnoreCase))
            {
                Channels.Add(new ChannelChoiceViewModel(name, wanted.Contains(name), OnChannelToggled));
            }

            foreach (var name in wanted)
            {
                ShownChannels.Add(Channels.First(c => c.Name == name));
            }
        }

        _suppressChannelEvents = false;
        OnPropertyChanged(nameof(HasChannels));
        OnPropertyChanged(nameof(HasShownChannels));
        if (log is null)
        {
            _fixture = null;
        }

        base.OnSessionLoaded();
    }

    private void OnChannelToggled(ChannelChoiceViewModel channel)
    {
        if (_suppressChannelEvents)
        {
            return;
        }

        if (channel.IsSelected && !ShownChannels.Contains(channel))
        {
            ShownChannels.Add(channel);
        }
        else if (!channel.IsSelected)
        {
            ShownChannels.Remove(channel);
        }

        OnShownChanged();
    }

    /// <summary>Move a plotted channel from one panel position to another.</summary>
    public void MoveShown(int from, int to)
    {
        if (from == to || from < 0 || to < 0 || from >= ShownChannels.Count || to >= ShownChannels.Count)
        {
            return;
        }

        ShownChannels.Move(from, to);
        OnShownChanged();
    }

    private void OnShownChanged()
    {
        OnPropertyChanged(nameof(HasShownChannels));
        if (_settings is not null && VehicleKey(_sessionA) is { } key)
        {
            _settings.ChannelsByVehicle[key] = ShownChannels.Select(c => c.Name).ToList();
        }

        OnSelectionChanged();
    }

    /// <summary>Everything one analysis reads, snapshotted on the UI thread.</summary>
    public sealed record Parameters(
        LogFile Log,
        List<int> Laps,
        List<string> Channels,
        string Timebase,
        Dictionary<int, LapStyle> Styles,
        string ViewKey);

    /// <inheritdoc />
    protected override Parameters? GetParameters()
    {
        if (_sessionA.Log is null)
        {
            return Hint(NoSessionHint);
        }

        var laps = _sessionA.SelectedIn(Scope);
        if (laps.Count == 0)
        {
            return Hint("Select laps to overlay");
        }

        var channels = ShownChannels.Select(c => c.Name).ToList();
        if (channels.Count == 0)
        {
            return Hint("Select channels to plot");
        }

        var styles = laps.Where(n => _sessionA.StyleOf(n) is not null).ToDictionary(n => n, n => _sessionA.StyleOf(n)!.Value);
        return new Parameters(_sessionA.Log, laps, channels, Timebase, styles, _sessionA.FilePath ?? string.Empty);
    }

    /// <inheritdoc />
    protected override Fixture Analyze(Parameters p, CancellationToken cancellation)
    {
        // Fastest selected lap first: it is the delta-time reference.
        var ordered = p.Laps.OrderBy(n => p.Log.Laps.First(l => l.Num == n).LapTimeMs).ToList();
        var fixture = LapSeriesBuilder.Build(p.Log, ordered, p.Channels, p.Timebase, cancellation);
        foreach (var lap in fixture.Laps)
        {
            lap.Style = p.Styles.TryGetValue(lap.Num, out var style) ? style : null;
        }

        return fixture;
    }

    /// <inheritdoc />
    protected override void Display(Parameters p, Fixture fixture)
    {
        _viewKey = p.ViewKey;
        Show(fixture);
    }

    /// <inheritdoc />
    protected override void ClearDisplay()
    {
        _fixture = null;
        Spec = null;
    }

    private void Show(Fixture fixture)
    {
        _fixture = fixture;
        Summary = $"{fixture.Laps.Count} laps, {fixture.SampleCount:N0} samples";
        Spec = StripChartBuilder.Build(fixture, XAxis, _viewKey);
    }

    /// <inheritdoc />
    public override void SaveState(IDictionary<string, string> state)
    {
        base.SaveState(state);
        state["x_axis"] = XAxis.ToString();
    }

    /// <inheritdoc />
    public override void RestoreState(IReadOnlyDictionary<string, string> state)
    {
        base.RestoreState(state);
        if (state.TryGetValue("x_axis", out var axis) && Enum.TryParse<XAxisMode>(axis, out var mode))
        {
            XAxis = mode;
        }
    }
}
