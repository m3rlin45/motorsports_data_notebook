using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.App.Series;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>A channel the strip chart can show, with its checkbox.</summary>
public sealed partial class ChannelChoiceViewModel : ObservableObject
{
    private readonly Action _onToggled;

    /// <summary>Create a choice for <paramref name="name"/>.</summary>
    public ChannelChoiceViewModel(string name, bool isSelected, Action onToggled)
    {
        Name = name;
        _isSelected = isSelected;
        _onToggled = onToggled;
    }

    /// <summary>Channel name.</summary>
    public string Name { get; }

    [ObservableProperty]
    private bool _isSelected;

    partial void OnIsSelectedChanged(bool value) => _onToggled();
}

/// <summary>
/// The Time/Distance tab: the selected laps of Session A overlaid as a
/// strip chart (one panel per chosen channel plus delta time), with the
/// linked crosshair. Also hosts the demo fixtures (`cs-run path.json`).
/// </summary>
public sealed partial class TimeDistanceViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _sessionA;
    private Fixture? _fixture;
    private bool _suppressChannelEvents;
    private List<string> _defaultOrder = new();

    /// <summary>Create the tab over <paramref name="sessionA"/>.</summary>
    public TimeDistanceViewModel(SessionViewModel sessionA)
    {
        _sessionA = sessionA;
    }

    /// <inheritdoc />
    public override string Header => "Time / Distance";

    /// <summary>The linked crosshair group the chart joins.</summary>
    public CursorLinkGroup Cursors { get; } = new();

    /// <summary>Channels available in Session A, with the plotted ones checked.</summary>
    public ObservableCollection<ChannelChoiceViewModel> Channels { get; } = new();

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

    partial void OnXAxisChanged(XAxisMode value)
    {
        Cursors.SetCursor(null);
        if (_fixture is not null)
        {
            Display();
        }
    }

    /// <summary>Show a dumped fixture instead of a session (demo mode).</summary>
    public void ShowFixture(Fixture fixture)
    {
        _fixture = fixture;
        Display();
        Status = "Fixture loaded";
    }

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        var log = _sessionA.Log;
        var profile = _sessionA.Profile;
        var names = profile?.ChannelNames ?? ChannelKeys.DefaultChannelNames;
        Timebase = names["gps_speed"];
        _defaultOrder = new List<string> { "speed_kmh", names["throttle"], names["brake"] };
        var defaults = new HashSet<string>(_defaultOrder);

        _suppressChannelEvents = true;
        Channels.Clear();
        if (log is not null)
        {
            foreach (var name in log.ChannelNames.OrderBy(n => n, StringComparer.OrdinalIgnoreCase))
            {
                Channels.Add(new ChannelChoiceViewModel(name, defaults.Contains(name), OnChannelToggled));
            }
        }

        _suppressChannelEvents = false;
        base.OnSessionLoaded();
    }

    private void OnChannelToggled()
    {
        if (!_suppressChannelEvents)
        {
            OnSelectionChanged();
        }
    }

    private sealed record Parameters(LogFile Log, List<int> Laps, List<string> Channels, string Timebase);

    /// <inheritdoc />
    protected override object? GetParameters()
    {
        if (_sessionA.Log is null)
        {
            return null;
        }

        var laps = _sessionA.SelectedLaps;
        if (laps.Count == 0)
        {
            Status = "Select laps to analyze";
            return null;
        }

        // Panels in the familiar order: speed, throttle, brake first, then the rest alphabetically.
        var selected = Channels.Where(c => c.IsSelected).Select(c => c.Name).ToList();
        var channels = _defaultOrder.Where(selected.Contains).Concat(selected.Where(n => !_defaultOrder.Contains(n))).ToList();
        if (channels.Count == 0)
        {
            Status = "Select channels to plot";
            return null;
        }

        return new Parameters(_sessionA.Log, laps, channels, Timebase);
    }

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        // Fastest selected lap first: it is the delta-time reference.
        var ordered = p.Laps.OrderBy(n => p.Log.Laps.First(l => l.Num == n).LapTimeMs).ToList();
        _fixture = LapSeriesBuilder.Build(p.Log, ordered, p.Channels, p.Timebase);
    }

    /// <inheritdoc />
    protected override void Display()
    {
        if (_fixture is null)
        {
            return;
        }

        Summary = $"{_fixture.Laps.Count} laps, {_fixture.SampleCount:N0} samples";
        Spec = StripChartBuilder.Build(_fixture, XAxis);
        ColourLapLists(_fixture);
    }

    /// <summary>
    /// Stamp each drawn lap's trace colour onto its row in the session panel,
    /// so the lap picker is the legend; laps not drawn lose their swatch.
    /// </summary>
    private void ColourLapLists(Fixture fixture)
    {
        foreach (var item in _sessionA.Laps)
        {
            item.TraceColor = null;
        }

        for (var i = 0; i < fixture.Laps.Count; i++)
        {
            var lap = fixture.Laps[i];
            var item = _sessionA.Laps.FirstOrDefault(l => l.Num == lap.Num);
            if (item is not null)
            {
                item.TraceColor = StripChartBuilder.ColorFor(i);
            }
        }
    }
}
