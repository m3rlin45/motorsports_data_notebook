using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Corner-detection and throttle-acceptance thresholds as validated text fields.</summary>
public sealed partial class DriverThresholdsViewModel : ObservableObject
{
    /// <summary>Python defaults.</summary>
    public const double DefaultCornerThreshold = 0.006;
    public const double DefaultThrottleThreshold = 98.0;
    public const double DefaultSustainMs = 500.0;

    /// <summary>Create with the defaults; <paramref name="onChanged"/> is called when the user edits a field.</summary>
    public DriverThresholdsViewModel(Action onChanged)
    {
        CornerThreshold = new NumericFieldViewModel("Corner detection threshold", "0.###", min: 0, minExclusive: true, onChanged: onChanged);
        ThrottleThreshold = new NumericFieldViewModel("Throttle threshold", "0.#", min: 0, max: 100, minExclusive: true, onChanged: onChanged);
        SustainMs = new NumericFieldViewModel("Sustain time", "0", min: 0, onChanged: onChanged);
        Reset();
    }

    /// <summary>Corner curvature threshold, 1/m.</summary>
    public NumericFieldViewModel CornerThreshold { get; }

    /// <summary>Full-throttle threshold, %.</summary>
    public NumericFieldViewModel ThrottleThreshold { get; }

    /// <summary>Sustain time, ms.</summary>
    public NumericFieldViewModel SustainMs { get; }

    /// <summary>The three fields.</summary>
    public IEnumerable<NumericFieldViewModel> Fields => new[] { CornerThreshold, ThrottleThreshold, SustainMs };

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    /// <summary>Restore the defaults silently.</summary>
    public void Reset()
    {
        CornerThreshold.Set(DefaultCornerThreshold);
        ThrottleThreshold.Set(DefaultThrottleThreshold);
        SustainMs.Set(DefaultSustainMs);
    }

    /// <summary>Copy from another instance silently.</summary>
    public void CopyFrom(DriverThresholdsViewModel other)
    {
        CornerThreshold.CopyFrom(other.CornerThreshold);
        ThrottleThreshold.CopyFrom(other.ThrottleThreshold);
        SustainMs.CopyFrom(other.SustainMs);
    }
}

/// <summary>Driver tab config: throttle/brake/lateral/GPS channels and thresholds per session.</summary>
public sealed class DriverConfigViewModel : ChannelConfigViewModel
{
    /// <summary>Create the config.</summary>
    public DriverConfigViewModel()
        : base(new[]
        {
            ("throttle", "Throttle"), ("brake", "Brake"), ("lateral_g", "Lateral G"),
            ("gps_latitude", "GPS Lat"), ("gps_longitude", "GPS Lon"), ("gps_speed", "GPS Speed"),
        })
    {
        ThresholdsA = new DriverThresholdsViewModel(() => OnThresholdsChanged(SessionSlot.A));
        ThresholdsB = new DriverThresholdsViewModel(() => OnThresholdsChanged(SessionSlot.B)) { IsEditable = false };
    }

    /// <summary>Session A thresholds.</summary>
    public DriverThresholdsViewModel ThresholdsA { get; }

    /// <summary>Session B thresholds.</summary>
    public DriverThresholdsViewModel ThresholdsB { get; }

    /// <summary>The thresholds being shown/edited.</summary>
    public DriverThresholdsViewModel Thresholds => ActiveSession == SessionSlot.B ? ThresholdsB : ThresholdsA;

    /// <summary>Raised when thresholds change.</summary>
    public event Action? ThresholdsChanged;

    /// <summary>Thresholds for a session (B follows A while synced).</summary>
    public DriverThresholdsViewModel GetThresholds(SessionSlot session) => session == SessionSlot.B && !SyncWithA ? ThresholdsB : ThresholdsA;

    private void OnThresholdsChanged(SessionSlot session)
    {
        if (session == SessionSlot.A && SyncWithA)
        {
            ThresholdsB.CopyFrom(ThresholdsA);
        }

        ThresholdsChanged?.Invoke();
    }

    /// <inheritdoc />
    protected override IEnumerable<NumericFieldViewModel> NumericFields(SessionSlot session) =>
        (session == SessionSlot.B ? ThresholdsB : ThresholdsA).Fields;

    /// <inheritdoc />
    protected override void CopyExtrasAToB() => ThresholdsB.CopyFrom(ThresholdsA);

    /// <inheritdoc />
    protected override void OnSyncChanged(bool synced) => ThresholdsB.IsEditable = !synced;

    /// <inheritdoc />
    protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.PropertyName == nameof(ActiveSession))
        {
            OnPropertyChanged(nameof(Thresholds));
        }
    }
}

/// <summary>A corner in the selector list.</summary>
public sealed record CornerChoice(int Index, string Label);

/// <summary>The three driver views.</summary>
public enum DriverViewMode
{
    /// <summary>Per-corner box plots.</summary>
    Summary,

    /// <summary>Multi-lap overlay of one corner.</summary>
    Detail,

    /// <summary>Track map with segments.</summary>
    Map,
}

/// <summary>Builds the driver figures and tables from <see cref="DriverConsistencyResult"/>s.</summary>
public static class DriverChartBuilder
{
    private const string Blue = ChartColors.SessionA;
    private const string Orange = ChartColors.SessionB;
    private const string Gold = ChartColors.Opportunity;

    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    private static string CornerLabel(Corner c) => $"{c.Name} ({c.Direction})";

    /// <summary>The <c>_box_stats_text</c> tooltip.</summary>
    public static string BoxStatsText(IReadOnlyList<double> values, string cornerName, string unit, string label = "")
    {
        var box = BoxSpec.From(0, values, Blue, string.Empty);
        var header = label.Length > 0 ? $"{cornerName} ({label})" : cornerName;
        return $"{header}\nMean {F(box.Mean, "0.00")}{unit}\nMedian {F(box.Median, "0.00")}{unit}\nQ1 {F(box.Q1, "0.00")}{unit} · Q3 {F(box.Q3, "0.00")}{unit}\nWhiskers {F(box.WhiskerLow, "0.00")} – {F(box.WhiskerHigh, "0.00")}{unit}\nN {values.Count}";
    }

    private static double[] Centered(List<double> values)
    {
        var mean = values.Average();
        return values.Select(v => v - mean).ToArray();
    }

    /// <summary><c>draw_summary</c>: BP offset, throttle acceptance and exit speed boxes per corner.</summary>
    public static BoxPlotSpec Summary(DriverConsistencyResult a, DriverConsistencyResult? b, string labelA, string labelB)
    {
        var corners = a.CornerData;
        var n = corners.Count;
        var positions = Enumerable.Range(1, n).Select(i => (double)i).ToArray();
        var labels = corners.Select(cd => CornerLabel(cd.Corner)).ToArray();
        var opp = corners.Select(cd => cd.OpportunityScore).ToArray();
        var maxOpp = opp.Any(s => s > 0) ? opp.Max() : 0.0;

        var bp = new List<BoxSpec>();
        var ta = new List<BoxSpec>();
        var exit = new List<BoxSpec>();
        void AddBoxes(IReadOnlyList<CornerConsistency> data, double offset, double width, string color, string label)
        {
            for (var i = 0; i < n; i++)
            {
                var cd = data.Count > i ? data[i] : null;
                if (cd is null)
                {
                    continue;
                }

                var pos = positions[i] + offset;
                if (cd.BpValues.Count > 0)
                {
                    bp.Add(BoxSpec.From(pos, Centered(cd.BpValues), color, BoxStatsText(Centered(cd.BpValues), labels[i], " m", label), width));
                }

                if (cd.TaValues.Count > 0)
                {
                    ta.Add(BoxSpec.From(pos, cd.TaValues, color, BoxStatsText(cd.TaValues, labels[i], "%", label), width));
                }

                if (cd.ExitSpeedValues.Count > 0)
                {
                    exit.Add(BoxSpec.From(pos, cd.ExitSpeedValues, color, BoxStatsText(cd.ExitSpeedValues, labels[i], " km/h", label), width));
                }
            }
        }

        string title;
        var legend = new List<(string, string)>();
        if (b is null)
        {
            // One session: every box is Session A's colour (orange would read as the comparison session);
            // the gold bands behind the columns carry the opportunity ranking.
            AddBoxes(corners, 0, 0.5, Blue, string.Empty);
            title = $"Driver Consistency: {labelA}";
        }
        else
        {
            // B's corners matched to A's by corner id, in A's order.
            var matched = corners.Select(cd => b.CornerData.FirstOrDefault(x => x.Corner.Id == cd.Corner.Id)).ToList();
            AddBoxes(corners, -0.2, 0.3, Blue, labelA);
            var bOrdered = matched.Select(m => m ?? new CornerConsistency { Corner = new Corner() }).ToList();
            AddBoxes(bOrdered, 0.2, 0.3, Orange, labelB);
            title = $"Comparison: {labelA} vs {labelB}";
            legend.Add((labelA, Blue));
            legend.Add((labelB, Orange));
        }

        // Gold bands behind the top-3 opportunity corners (single session only).
        var bands = new List<BandSpec>();
        if (b is null && maxOpp > 0)
        {
            var ranked = Enumerable.Range(0, n).Where(i => opp[i] > 0).OrderByDescending(i => opp[i]).Take(3).ToList();
            var alphas = new[] { 0.15, 0.10, 0.06 };
            for (var rank = 0; rank < ranked.Count; rank++)
            {
                bands.Add(new BandSpec(positions[ranked[rank]] - 0.45, positions[ranked[rank]] + 0.45, Gold, alphas[rank]));
            }
        }

        var panels = new[]
        {
            new BoxPanelSpec("Braking point offset", "BP Offset (m)", bp, ZeroLine: true),
            new BoxPanelSpec("Throttle acceptance", "TA (%)", ta),
            new BoxPanelSpec("Exit speed", "Exit Speed (km/h)", exit),
        };
        return new BoxPlotSpec(title, panels, positions, labels, bands, legend);
    }

    /// <summary>
    /// <c>draw_detail</c>: throttle, brake and lateral G of every lap through
    /// one corner, with the brake/entry/apex/exit lines. Each lap is drawn in
    /// the style its row of the lap list shows (<paramref name="styles"/>),
    /// so the list is the legend; <paramref name="viewKey"/> keeps the zoom
    /// while the same corner is re-analysed.
    /// </summary>
    public static StripChartSpec Detail(CornerConsistency cd, IReadOnlyDictionary<int, LapStyle>? styles = null, string? viewKey = null)
    {
        var traces = cd.LapTraces;
        TraceSpec Trace(LapTrace t, int i, double[] y)
        {
            var style = styles is not null && styles.TryGetValue(t.LapNum, out var s) ? s : new LapStyle(i % LapStyleAllocator.SlotCount);
            return new TraceSpec($"Lap {t.LapNum}", t.Distance, y, style.Color, Dashed: style.Dashed, ShortLabel: $"L{t.LapNum}");
        }

        var throttle = traces.Select((t, i) => Trace(t, i, t.Throttle)).ToList();
        var brake = traces.Select((t, i) => Trace(t, i, t.Brake)).ToList();
        var lateral = traces.Select((t, i) => Trace(t, i, t.LateralG)).ToList();

        var markers = new List<XMarkerSpec>();
        if (cd.BrakingStart is { } bs)
        {
            markers.Add(new XMarkerSpec(bs, "Brake", ChartColors.MarkerBrake, 0));
        }

        markers.Add(new XMarkerSpec(cd.Corner.StartDist, "Entry", ChartColors.MarkerCorner, 1));
        markers.Add(new XMarkerSpec(cd.Corner.ApexDist, "Apex", ChartColors.MarkerApex, 2));
        markers.Add(new XMarkerSpec(cd.Corner.EndDist, "Exit", ChartColors.MarkerCorner, 3));

        var taNote = cd.TaValues.Count > 0 ? $" — TA {F(cd.TaMean, "0.0")}% (std {F(cd.TaStd, "0.0")}%)" : string.Empty;
        var panels = new[]
        {
            new SubplotSpec($"Throttle position{taNote}", "%", throttle),
            new SubplotSpec("Brake pressure", string.Empty, brake),
            new SubplotSpec("Lateral G", "g", lateral),
        };
        var start = traces.Count > 0 ? traces.Min(t => t.Distance.First()) : cd.Corner.StartDist - 150;
        var end = traces.Count > 0 ? traces.Max(t => t.Distance.Last()) : cd.Corner.EndDist + 50;
        return new StripChartSpec(panels, XAxisMode.Distance, markers, new XRange(start, end), Legend: false, ViewKey: viewKey);
    }

    /// <summary>
    /// <c>draw_track_map</c>: the reference lap as a neutral outline with the
    /// braking and acceleration zones coloured over it, apex labels (clickable:
    /// their id is the corner's index) and gold stars on the top-3
    /// opportunity corners.
    /// </summary>
    public static TrackMapSpec Map(DriverConsistencyResult r)
    {
        var (x, y) = Corners.GpsToLocalXy(r.RefLat, r.RefLon);
        var distance = r.RefDistance;
        // Corners are the outline itself: only what the driver does about them is coloured.
        var paths = new List<TrackMapPathSpec> { new(x, y, ChartColors.Neutral, 2) };
        var colors = new Dictionary<string, string> { ["braking"] = ChartColors.Braking, ["acceleration"] = ChartColors.Acceleration };
        var names = new Dictionary<string, string> { ["braking"] = "Braking", ["acceleration"] = "Acceleration" };
        var opp = r.CornerData.Where(cd => cd.OpportunityScore > 0).ToDictionary(cd => cd.Corner.Id, cd => cd.OpportunityScore);
        var maxOpp = opp.Count > 0 ? opp.Values.Max() : 0.0;
        var legendDone = new HashSet<string>();

        foreach (var seg in r.Segments)
        {
            if (!colors.TryGetValue(seg.SegmentType, out var color))
            {
                continue;
            }

            var idx = Enumerable.Range(0, distance.Length).Where(i => distance[i] >= seg.StartDist && distance[i] <= seg.EndDist).ToArray();
            if (idx.Length == 0)
            {
                continue;
            }

            var width = 4.0;
            if (seg.SegmentType == "acceleration" && opp.TryGetValue(seg.CornerId, out var score) && maxOpp > 0)
            {
                width = 4 + 6 * score / maxOpp;
            }

            var label = legendDone.Add(seg.SegmentType) ? names.GetValueOrDefault(seg.SegmentType, seg.SegmentType) : string.Empty;
            paths.Add(new TrackMapPathSpec(idx.Select(i => x[i]).ToArray(), idx.Select(i => y[i]).ToArray(), color, width, label));
        }

        var points = new List<TrackMapPointSpec>();
        var topIds = opp.OrderByDescending(kv => kv.Value).Take(3).Select(kv => kv.Key).ToHashSet();
        foreach (var seg in r.Segments.Where(s => s.SegmentType == "corner" && s.ApexDist is not null))
        {
            var apex = Numerics.ArgMin(distance.Select(d => Math.Abs(d - seg.ApexDist!.Value)).ToArray());
            if (topIds.Contains(seg.CornerId))
            {
                points.Add(new TrackMapPointSpec(x[apex], y[apex], Gold, 16, Star: true));
            }

            var cornerIndex = r.Corners.ToList().FindIndex(c => c.Id == seg.CornerId);
            points.Add(new TrackMapPointSpec(x[apex], y[apex], ChartColors.Ink, 7, seg.Name, Id: cornerIndex));
        }

        return new TrackMapSpec("Click a corner to see its laps", paths, points);
    }

    /// <summary>The stats panel tables: entry consistency and corner speed / exit.</summary>
    public static List<StatsTable> Tables(DriverConsistencyResult a, DriverConsistencyResult? b)
    {
        static string Opt(bool has, double v) => has ? v.ToString("0.0", CultureInfo.InvariantCulture) : "N/A";
        CornerConsistency? Match(CornerConsistency cd) => b?.CornerData.FirstOrDefault(x => x.Corner.Id == cd.Corner.Id);

        StatsTable entry;
        StatsTable speed;
        if (b is null)
        {
            entry = new StatsTable("Entry Consistency", new[] { "Corner", "TA Mean (%)", "TA Std", "BP Std (m)", "Laps" },
                a.CornerData.Select(cd => (IReadOnlyList<string>)new[] { CornerLabel(cd.Corner), Opt(cd.TaValues.Count > 0, cd.TaMean), Opt(cd.TaValues.Count > 0, cd.TaStd), Opt(cd.BpValues.Count > 0, cd.BpStd), cd.LapTraces.Count.ToString(CultureInfo.InvariantCulture) }).ToList());
            speed = new StatsTable("Corner Speed & Exit", new[] { "Corner", "Min Spd", "Min Std", "Exit Spd", "Exit Std", "Laps" },
                a.CornerData.Select(cd => (IReadOnlyList<string>)new[] { CornerLabel(cd.Corner), Opt(cd.SpeedValues.Count > 0, cd.SpeedMean), Opt(cd.SpeedValues.Count > 0, cd.SpeedStd), Opt(cd.ExitSpeedValues.Count > 0, cd.ExitSpeedMean), Opt(cd.ExitSpeedValues.Count > 0, cd.ExitSpeedStd), cd.LapTraces.Count.ToString(CultureInfo.InvariantCulture) }).ToList());
        }
        else
        {
            entry = new StatsTable("Entry Consistency", new[] { "Corner", "A TA Mean", "A TA Std", "B TA Mean", "B TA Std", "A BP Std", "B BP Std" },
                a.CornerData.Select(cd =>
                {
                    var m = Match(cd);
                    return (IReadOnlyList<string>)new[] { CornerLabel(cd.Corner), Opt(cd.TaValues.Count > 0, cd.TaMean), Opt(cd.TaValues.Count > 0, cd.TaStd), Opt(m is { TaValues.Count: > 0 }, m?.TaMean ?? 0), Opt(m is { TaValues.Count: > 0 }, m?.TaStd ?? 0), Opt(cd.BpValues.Count > 0, cd.BpStd), Opt(m is { BpValues.Count: > 0 }, m?.BpStd ?? 0) };
                }).ToList());
            speed = new StatsTable("Corner Speed & Exit", new[] { "Corner", "A Min", "A Min Std", "B Min", "B Min Std", "A Exit", "A Exit Std", "B Exit", "B Exit Std" },
                a.CornerData.Select(cd =>
                {
                    var m = Match(cd);
                    return (IReadOnlyList<string>)new[] { CornerLabel(cd.Corner), Opt(cd.SpeedValues.Count > 0, cd.SpeedMean), Opt(cd.SpeedValues.Count > 0, cd.SpeedStd), Opt(m is { SpeedValues.Count: > 0 }, m?.SpeedMean ?? 0), Opt(m is { SpeedValues.Count: > 0 }, m?.SpeedStd ?? 0), Opt(cd.ExitSpeedValues.Count > 0, cd.ExitSpeedMean), Opt(cd.ExitSpeedValues.Count > 0, cd.ExitSpeedStd), Opt(m is { ExitSpeedValues.Count: > 0 }, m?.ExitSpeedMean ?? 0), Opt(m is { ExitSpeedValues.Count: > 0 }, m?.ExitSpeedStd ?? 0) };
                }).ToList());
        }

        return new List<StatsTable> { entry, speed };
    }
}

/// <summary>The Driver Consistency tab.</summary>
public sealed partial class DriverViewModel : AnalysisTabViewModel<DriverViewModel.Parameters, DriverViewModel.Result>
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private Result? _result;
    private Parameters? _shown;
    private string? _autoThrottleChannelA;
    private bool _autoThrottleA;
    private bool _autoThrottleB;
    private bool _keepViewMode;

    /// <summary>Create the tab over the two session slots.</summary>
    public DriverViewModel(SessionViewModel a, SessionViewModel b)
    {
        _a = a;
        _b = b;
        DriverConfig = new DriverConfigViewModel();
        DriverConfig.Changed += OnConfigChanged;
        DriverConfig.ThresholdsChanged += OnSelectionChanged;
        Commands = new[]
        {
            new TabCommand("Back to _summary", "Escape", ShowSummaryCommand),
        };
    }

    /// <inheritdoc />
    public override string Header => "Driver Consistency";

    /// <summary>Channels and thresholds.</summary>
    public DriverConfigViewModel DriverConfig { get; }

    /// <inheritdoc />
    public override object? Config => DriverConfig;

    /// <inheritdoc />
    public override bool HasStats => true;

    /// <inheritdoc />
    public override bool UsesLapColors => ViewMode == DriverViewMode.Detail;

    /// <inheritdoc />
    public override IReadOnlyList<TabCommand> Commands { get; }

    /// <summary>Corners of the current result, for the selector.</summary>
    public ObservableCollection<CornerChoice> Corners { get; } = new();

    /// <summary>Which figure is shown.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSummary))]
    [NotifyPropertyChangedFor(nameof(IsDetail))]
    [NotifyPropertyChangedFor(nameof(IsMap))]
    [NotifyPropertyChangedFor(nameof(UsesLapColors))]
    private DriverViewMode _viewMode = DriverViewMode.Summary;

    /// <summary>Radio binding.</summary>
    public bool IsSummary
    {
        get => ViewMode == DriverViewMode.Summary;
        set
        {
            if (value)
            {
                ViewMode = DriverViewMode.Summary;
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsDetail
    {
        get => ViewMode == DriverViewMode.Detail;
        set
        {
            if (value)
            {
                ViewMode = DriverViewMode.Detail;
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsMap
    {
        get => ViewMode == DriverViewMode.Map;
        set
        {
            if (value)
            {
                ViewMode = DriverViewMode.Map;
            }
        }
    }

    /// <summary>Return to the summary (Esc).</summary>
    [RelayCommand]
    public void ShowSummary() => ViewMode = DriverViewMode.Summary;

    /// <summary>The corner shown in the detail view.</summary>
    [ObservableProperty]
    private CornerChoice? _selectedCorner;

    partial void OnSelectedCornerChanged(CornerChoice? value)
    {
        if (value is not null && !_keepViewMode)
        {
            ViewMode = DriverViewMode.Detail; // selecting a corner switches to the detail view, as in Python
        }

        UpdateDetail();
    }

    /// <summary>
    /// Show the laps through corner <paramref name="index"/>: what a click
    /// on its box in the summary, its apex on the map or its row in the list
    /// does. Works for the corner that is already selected too.
    /// </summary>
    public void OpenCorner(int index)
    {
        if (index < 0 || index >= Corners.Count)
        {
            return;
        }

        SelectedCorner = Corners[index];
        ViewMode = DriverViewMode.Detail;
    }

    /// <summary>Box plots.</summary>
    [ObservableProperty]
    private BoxPlotSpec? _summarySpec;

    /// <summary>Overlay for the selected corner.</summary>
    [ObservableProperty]
    private StripChartSpec? _detailSpec;

    /// <summary>Track map.</summary>
    [ObservableProperty]
    private TrackMapSpec? _mapSpec;

    /// <summary>Stats tables.</summary>
    [ObservableProperty]
    private IReadOnlyList<StatsTable> _stats = Array.Empty<StatsTable>();

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        DriverConfig.OnSessionsChanged(_a, _b);
        // The throttle threshold follows each session's own throttle range; it is derived with the next analysis.
        _autoThrottleA = _a.Log is not null;
        _autoThrottleB = _b.Log is not null;
        _autoThrottleChannelA = DriverConfig.GetChannelNames(SessionSlot.A)["throttle"];
        base.OnSessionLoaded();
    }

    private void OnConfigChanged()
    {
        // A throttle channel change re-derives the auto threshold, as in Python.
        var current = DriverConfig.GetChannelNames(SessionSlot.A)["throttle"];
        if (_autoThrottleChannelA is not null && current != _autoThrottleChannelA)
        {
            _autoThrottleChannelA = current;
            _autoThrottleA = true;
        }

        OnSelectionChanged();
    }

    /// <summary>
    /// Throttle threshold = 95 % of the channel's 95th percentile, rounded to
    /// 0.1 (<c>_auto_set_throttle_threshold</c>); null when the channel is
    /// missing or never opens.
    /// </summary>
    public static double? AutoThrottleThreshold(LogFile log, string throttleChannel)
    {
        if (!log.TryGetChannel(throttleChannel, out var channel) || channel.Length == 0)
        {
            return null;
        }

        var peak = Numerics.Percentile(channel.Values.Where(v => !double.IsNaN(v)).ToArray(), 95);
        return peak <= 0 ? null : Math.Round(peak * 0.95, 1, MidpointRounding.ToEven);
    }

    /// <summary>One session's inputs.</summary>
    public sealed record SessionInputs(
        LogFile Log,
        List<int> Laps,
        Dictionary<string, string> Names,
        double CornerThreshold,
        double ThrottleThreshold,
        bool AutoThrottle,
        double SustainMs,
        string Label,
        string? Path,
        IReadOnlyDictionary<int, LapStyle> Styles);

    /// <summary>
    /// Everything one analysis reads, snapshotted on the UI thread.
    /// <paramref name="BFollowsA"/>: B shares A's thresholds, the derived
    /// throttle threshold included.
    /// </summary>
    public sealed record Parameters(SessionInputs A, SessionInputs? B, bool BFollowsA);

    /// <summary>The results, and the throttle thresholds they were computed with (derived ones included).</summary>
    public sealed record Result(DriverConsistencyResult A, double ThrottleA, DriverConsistencyResult? B, double ThrottleB);

    private SessionInputs Inputs(SessionViewModel session, SessionSlot slot, List<int> laps, bool autoThrottle)
    {
        var t = DriverConfig.GetThresholds(slot);
        return new SessionInputs(
            session.Log!,
            laps,
            DriverConfig.GetChannelNames(slot),
            t.CornerThreshold.Value ?? DriverThresholdsViewModel.DefaultCornerThreshold,
            t.ThrottleThreshold.Value ?? DriverThresholdsViewModel.DefaultThrottleThreshold,
            autoThrottle,
            t.SustainMs.Value ?? DriverThresholdsViewModel.DefaultSustainMs,
            session.SessionLabel,
            session.FilePath,
            laps.Where(n => session.StyleOf(n) is not null).ToDictionary(n => n, n => session.StyleOf(n)!.Value));
    }

    /// <inheritdoc />
    protected override Parameters? GetParameters()
    {
        if (_a.Log is null)
        {
            return Hint(NoSessionHint);
        }

        var lapsA = _a.SelectedIn(Scope);
        if (lapsA.Count == 0)
        {
            return Hint("Select laps to analyze");
        }

        // A comparison session with no laps ticked is simply not compared.
        var lapsB = _b.Log is null ? new List<int>() : _b.SelectedIn(Scope);
        var compare = lapsB.Count > 0;
        if ((DriverConfig.FirstError(SessionSlot.A) ?? (compare ? DriverConfig.FirstError(SessionSlot.B) : null)) is { } error)
        {
            return Hint(error);
        }

        var follows = DriverConfig.SyncWithA;
        return new Parameters(
            Inputs(_a, SessionSlot.A, lapsA, _autoThrottleA),
            compare ? Inputs(_b, SessionSlot.B, lapsB, _autoThrottleB && !follows) : null,
            follows);
    }

    /// <inheritdoc />
    protected override Result Analyze(Parameters p, CancellationToken cancellation)
    {
        static (DriverConsistencyResult Result, double Throttle) Run(SessionInputs s, CancellationToken cancellation)
        {
            var throttle = s.AutoThrottle && AutoThrottleThreshold(s.Log, s.Names["throttle"]) is { } auto ? auto : s.ThrottleThreshold;
            return (DriverConsistency.Analyze(s.Log, s.Laps, s.Names, s.CornerThreshold, throttle, s.SustainMs, cancellation), throttle);
        }

        var a = Run(p.A, cancellation);
        if (p.B is null)
        {
            return new Result(a.Result, a.Throttle, null, a.Throttle);
        }

        var b = Run(p.BFollowsA ? p.B with { ThrottleThreshold = a.Throttle, AutoThrottle = false } : p.B, cancellation);
        return new Result(a.Result, a.Throttle, b.Result, b.Throttle);
    }

    /// <inheritdoc />
    protected override void Display(Parameters p, Result result)
    {
        _result = result;
        _shown = p;

        // Show the thresholds the analysis derived, without triggering another run.
        if (p.A.AutoThrottle)
        {
            _autoThrottleA = false;
            DriverConfig.ThresholdsA.ThrottleThreshold.Set(result.ThrottleA);
            if (DriverConfig.SyncWithA)
            {
                DriverConfig.ThresholdsB.CopyFrom(DriverConfig.ThresholdsA);
            }
        }

        if (p.B is { AutoThrottle: true })
        {
            _autoThrottleB = false;
            DriverConfig.ThresholdsB.ThrottleThreshold.Set(result.ThrottleB);
        }

        var previous = SelectedCorner?.Index ?? 0;
        Corners.Clear();
        for (var i = 0; i < result.A.Corners.Count; i++)
        {
            Corners.Add(new CornerChoice(i, $"{result.A.Corners[i].Name} ({result.A.Corners[i].Direction})"));
        }

        SummarySpec = DriverChartBuilder.Summary(result.A, result.B, p.A.Label, p.B?.Label ?? string.Empty);
        MapSpec = DriverChartBuilder.Map(result.A);
        Stats = DriverChartBuilder.Tables(result.A, result.B);
        var keep = Corners.Count > 0 ? Corners[Math.Min(previous, Corners.Count - 1)] : null;
        if (!Equals(SelectedCorner, keep))
        {
            // Re-selecting after a rerun keeps the current view mode.
            _keepViewMode = true;
            SelectedCorner = keep;
            _keepViewMode = false;
        }

        UpdateDetail();

        var totalTa = result.A.CornerData.Sum(cd => cd.TaValues.Count);
        var threshold = result.ThrottleA.ToString("0.#", CultureInfo.InvariantCulture);
        SetStatus(
            totalTa > 0
                ? $"Done - {result.A.Corners.Count} corners, {totalTa} TA pts (thr: {threshold})"
                : $"Done - {result.A.Corners.Count} corners, 0 TA - {DiagnoseEmptyTa(p.A, threshold)}",
            totalTa > 0 ? StatusKind.Done : StatusKind.Hint);
    }

    /// <inheritdoc />
    protected override void ClearDisplay()
    {
        _result = null;
        _shown = null;
        SummarySpec = null;
        DetailSpec = null;
        MapSpec = null;
        Stats = Array.Empty<StatsTable>();
        Corners.Clear();
    }

    private static string DiagnoseEmptyTa(SessionInputs a, string threshold)
    {
        var lateralName = a.Names["lateral_g"];
        if (!a.Log.TryGetChannel(lateralName, out var lateral))
        {
            return $"Lat G channel '{lateralName}' not found";
        }

        if (lateral.Length > 0 && lateral.Values.Max(Math.Abs) < 0.1)
        {
            return "Lat G values near zero — check channel name";
        }

        return $"try lower threshold (currently {threshold})";
    }

    private void UpdateDetail()
    {
        if (_result is null || _shown is null || SelectedCorner is null || SelectedCorner.Index >= _result.A.CornerData.Count)
        {
            DetailSpec = null;
            return;
        }

        // The same corner of the same session keeps its zoom when laps or thresholds change.
        var viewKey = $"{_shown.A.Path}|corner {SelectedCorner.Index}";
        DetailSpec = DriverChartBuilder.Detail(_result.A.CornerData[SelectedCorner.Index], _shown.A.Styles, viewKey);
    }
}
