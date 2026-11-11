using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Corner-detection and throttle-acceptance thresholds as editable text.</summary>
public sealed partial class DriverThresholdsViewModel : ObservableObject
{
    /// <summary>Python defaults.</summary>
    public const double DefaultCornerThreshold = 0.006;
    public const double DefaultThrottleThreshold = 98.0;
    public const double DefaultSustainMs = 500.0;

    private readonly Action _onChanged;
    private bool _suppress;

    /// <summary>Create with the defaults.</summary>
    public DriverThresholdsViewModel(Action onChanged)
    {
        _onChanged = onChanged;
        Reset();
    }

    [ObservableProperty]
    private string _cornerThreshold = string.Empty;

    [ObservableProperty]
    private string _throttleThreshold = string.Empty;

    [ObservableProperty]
    private string _sustainMs = string.Empty;

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    partial void OnCornerThresholdChanged(string value) => Raise();
    partial void OnThrottleThresholdChanged(string value) => Raise();
    partial void OnSustainMsChanged(string value) => Raise();

    private void Raise()
    {
        if (!_suppress)
        {
            _onChanged();
        }
    }

    private static double Parse(string text, double fallback) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    /// <summary>Corner curvature threshold, 1/m.</summary>
    public double CornerThresholdValue => Parse(CornerThreshold, DefaultCornerThreshold);

    /// <summary>Full-throttle threshold, %.</summary>
    public double ThrottleThresholdValue => Parse(ThrottleThreshold, DefaultThrottleThreshold);

    /// <summary>Sustain time, ms.</summary>
    public double SustainMsValue => Parse(SustainMs, DefaultSustainMs);

    /// <summary>Restore the defaults silently.</summary>
    public void Reset()
    {
        _suppress = true;
        CornerThreshold = DefaultCornerThreshold.ToString("0.###", CultureInfo.InvariantCulture);
        ThrottleThreshold = DefaultThrottleThreshold.ToString("0.#", CultureInfo.InvariantCulture);
        SustainMs = DefaultSustainMs.ToString("0", CultureInfo.InvariantCulture);
        _suppress = false;
    }

    /// <summary>Set the throttle threshold silently (the auto-threshold on session load).</summary>
    public void SetThrottleSilently(double value)
    {
        _suppress = true;
        ThrottleThreshold = value.ToString("0.#", CultureInfo.InvariantCulture);
        _suppress = false;
    }

    /// <summary>Copy from another instance silently.</summary>
    public void CopyFrom(DriverThresholdsViewModel other)
    {
        _suppress = true;
        CornerThreshold = other.CornerThreshold;
        ThrottleThreshold = other.ThrottleThreshold;
        SustainMs = other.SustainMs;
        _suppress = false;
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
        ThresholdsA = new DriverThresholdsViewModel(() => OnThresholdsChanged("A"));
        ThresholdsB = new DriverThresholdsViewModel(() => OnThresholdsChanged("B")) { IsEditable = false };
    }

    /// <summary>Session A thresholds.</summary>
    public DriverThresholdsViewModel ThresholdsA { get; }

    /// <summary>Session B thresholds.</summary>
    public DriverThresholdsViewModel ThresholdsB { get; }

    /// <summary>The thresholds being shown/edited.</summary>
    public DriverThresholdsViewModel Thresholds => ActiveSession == "B" ? ThresholdsB : ThresholdsA;

    /// <summary>Raised when thresholds change.</summary>
    public event Action? ThresholdsChanged;

    /// <summary>Thresholds for a session (B follows A while synced).</summary>
    public DriverThresholdsViewModel GetThresholds(string session) => session == "B" && !SyncWithA ? ThresholdsB : ThresholdsA;

    private void OnThresholdsChanged(string session)
    {
        if (session == "A" && SyncWithA)
        {
            ThresholdsB.CopyFrom(ThresholdsA);
        }

        ThresholdsChanged?.Invoke();
    }

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
    private const string Blue = "#4682b4";
    private const string Orange = "#ff8c00";
    private const string Gold = "#ffd700";

    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    private static string CornerLabel(Corner c) => $"{c.Name} ({c.Direction})";

    /// <summary>The <c>_box_stats_text</c> tooltip.</summary>
    public static string BoxStatsText(IReadOnlyList<double> values, string cornerName, string unit, string label = "")
    {
        var box = BoxSpec.From(0, values, Blue, string.Empty);
        var header = label.Length > 0 ? $"{cornerName} ({label})" : cornerName;
        return $"{header}\n  Mean:    {F(box.Mean, "0.00")}{unit}\n  Median:  {F(box.Median, "0.00")}{unit}\n  Q1:      {F(box.Q1, "0.00")}{unit}\n  Q3:      {F(box.Q3, "0.00")}{unit}\n  Whisker: {F(box.WhiskerLow, "0.00")} – {F(box.WhiskerHigh, "0.00")}{unit}\n  N:       {values.Count}";
    }

    private static double[] Centered(List<double> values)
    {
        var mean = values.Average();
        return values.Select(v => v - mean).ToArray();
    }

    /// <summary>Opportunity colour between steelblue and goldenrod.</summary>
    private static string OpportunityColor(double norm)
    {
        static int Mix(int a, int b, double t) => (int)Math.Round(a + (b - a) * t);
        var r = Mix(0x46, 0xda, norm);
        var g = Mix(0x82, 0xa5, norm);
        var b = Mix(0xb4, 0x20, norm);
        return $"#{r:x2}{g:x2}{b:x2}";
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
        void AddBoxes(IReadOnlyList<CornerConsistency> data, double offset, double width, string color, string label, bool opportunityColors)
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
                    var c = opportunityColors && maxOpp > 0 && opp[i] > 0 ? OpportunityColor(opp[i] / maxOpp) : color;
                    exit.Add(BoxSpec.From(pos, cd.ExitSpeedValues, c, BoxStatsText(cd.ExitSpeedValues, labels[i], " km/h", label), width));
                }
            }
        }

        string title;
        var legend = new List<(string, string)>();
        if (b is null)
        {
            AddBoxes(corners, 0, 0.5, Orange, string.Empty, opportunityColors: false);
            // Python colours BP orange, TA steelblue, exit by opportunity; redo TA/exit colours.
            ta = ta.Select(x => x with { Color = Blue }).ToList();
            exit = corners.Where(cd => cd.ExitSpeedValues.Count > 0).Select((cd, k) =>
            {
                var i = corners.IndexOf(cd);
                var c = maxOpp > 0 && opp[i] > 0 ? OpportunityColor(opp[i] / maxOpp) : Blue;
                return BoxSpec.From(positions[i], cd.ExitSpeedValues, c, BoxStatsText(cd.ExitSpeedValues, labels[i], " km/h"), 0.5);
            }).ToList();
            title = $"Driver Consistency: {labelA}";
        }
        else
        {
            // B's corners matched to A's by corner id, in A's order.
            var matched = corners.Select(cd => b.CornerData.FirstOrDefault(x => x.Corner.Id == cd.Corner.Id)).ToList();
            AddBoxes(corners, -0.2, 0.3, Blue, labelA, opportunityColors: false);
            var bOrdered = matched.Select(m => m ?? new CornerConsistency { Corner = new Corner() }).ToList();
            AddBoxes(bOrdered, 0.2, 0.3, Orange, labelB, opportunityColors: false);
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

    /// <summary>Viridis-like colour for lap i of n.</summary>
    private static string LapColor(int i, int n)
    {
        var t = n <= 1 ? 0.0 : (double)i / (n - 1);
        var stops = new[] { (0x44, 0x01, 0x54), (0x3b, 0x52, 0x8b), (0x21, 0x90, 0x8c), (0x5d, 0xc8, 0x63), (0xfd, 0xe7, 0x25) };
        var x = t * (stops.Length - 1);
        var k = Math.Min((int)Math.Floor(x), stops.Length - 2);
        var f = x - k;
        int Mix(int a, int b) => (int)Math.Round(a + (b - a) * f);
        return $"#{Mix(stops[k].Item1, stops[k + 1].Item1):x2}{Mix(stops[k].Item2, stops[k + 1].Item2):x2}{Mix(stops[k].Item3, stops[k + 1].Item3):x2}";
    }

    /// <summary><c>draw_detail</c>: throttle, brake and lateral G of every lap through one corner, with the brake/entry/apex/exit lines.</summary>
    public static StripChartSpec Detail(CornerConsistency cd)
    {
        var traces = cd.LapTraces;
        var n = traces.Count;
        TraceSpec Trace(LapTrace t, int i, double[] y) => new($"Lap {t.LapNum}", t.Distance, y, LapColor(i, n));
        var throttle = traces.Select((t, i) => Trace(t, i, t.Throttle)).ToList();
        var brake = traces.Select((t, i) => Trace(t, i, t.Brake)).ToList();
        var lateral = traces.Select((t, i) => Trace(t, i, t.LateralG)).ToList();

        var markers = new List<XMarkerSpec>();
        if (cd.BrakingStart is { } bs)
        {
            markers.Add(new XMarkerSpec(bs, "Brake", "#00a8b8", 0));
        }

        // Readable on both themes (Python's pure yellow/cyan are dark-theme colours).
        markers.Add(new XMarkerSpec(cd.Corner.StartDist, "Entry", "#d4a800", 1));
        markers.Add(new XMarkerSpec(cd.Corner.ApexDist, "Apex", "#e03030", 2));
        markers.Add(new XMarkerSpec(cd.Corner.EndDist, "Exit", "#d4a800", 3));

        var taNote = cd.TaValues.Count > 0 ? $" — TA {F(cd.TaMean, "0.0")}% (std {F(cd.TaStd, "0.0")}%)" : string.Empty;
        var panels = new[]
        {
            new SubplotSpec($"Throttle Position{taNote}", "%", throttle),
            new SubplotSpec("Brake Pressure", string.Empty, brake),
            new SubplotSpec("Lateral G", "g", lateral),
        };
        var start = traces.Count > 0 ? traces.Min(t => t.Distance.First()) : cd.Corner.StartDist - 150;
        var end = traces.Count > 0 ? traces.Max(t => t.Distance.Last()) : cd.Corner.EndDist + 50;
        return new StripChartSpec(panels, XAxisMode.Distance, markers, new XRange(start, end));
    }

    /// <summary><c>draw_track_map</c>: the reference lap with coloured segments, apex labels and gold stars on the top-3 opportunity corners.</summary>
    public static TrackMapSpec Map(DriverConsistencyResult r)
    {
        var (x, y) = Corners.GpsToLocalXy(r.RefLat, r.RefLon);
        var distance = r.RefDistance;
        var paths = new List<TrackMapPathSpec> { new(x, y, "#444444", 2) };
        var colors = new Dictionary<string, string> { ["braking"] = "#ff4444", ["corner"] = "#ff8800", ["acceleration"] = "#44bb44" };
        var names = new Dictionary<string, string> { ["braking"] = "Braking", ["corner"] = "Corner", ["acceleration"] = "Acceleration" };
        var opp = r.CornerData.Where(cd => cd.OpportunityScore > 0).ToDictionary(cd => cd.Corner.Id, cd => cd.OpportunityScore);
        var maxOpp = opp.Count > 0 ? opp.Values.Max() : 0.0;
        var legendDone = new HashSet<string>();

        foreach (var seg in r.Segments)
        {
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
            paths.Add(new TrackMapPathSpec(idx.Select(i => x[i]).ToArray(), idx.Select(i => y[i]).ToArray(), colors.GetValueOrDefault(seg.SegmentType, "#808080"), width, label));
        }

        var points = new List<TrackMapPointSpec>();
        var topIds = opp.OrderByDescending(kv => kv.Value).Take(3).Select(kv => kv.Key).ToHashSet();
        foreach (var seg in r.Segments.Where(s => s.SegmentType == "corner" && s.ApexDist is not null))
        {
            var apex = Numerics.ArgMin(distance.Select(d => Math.Abs(d - seg.ApexDist!.Value)).ToArray());
            points.Add(new TrackMapPointSpec(x[apex], y[apex], "#8b0000", 6, seg.Name));
            if (topIds.Contains(seg.CornerId))
            {
                points.Add(new TrackMapPointSpec(x[apex], y[apex], Gold, 14, Star: true));
            }
        }

        return new TrackMapSpec("Track Map", paths, points);
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
public sealed partial class DriverViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private DriverConsistencyResult? _resultA;
    private DriverConsistencyResult? _resultB;
    private string? _lastThrottleChannel;
    private bool _keepViewMode;

    /// <summary>Create the tab over the two session slots.</summary>
    public DriverViewModel(SessionViewModel a, SessionViewModel b)
    {
        _a = a;
        _b = b;
        DriverConfig = new DriverConfigViewModel();
        DriverConfig.Changed += OnConfigChanged;
        DriverConfig.ThresholdsChanged += OnSelectionChanged;
    }

    /// <inheritdoc />
    public override string Header => "Driver Consistency";

    /// <summary>Channels and thresholds.</summary>
    public DriverConfigViewModel DriverConfig { get; }

    /// <inheritdoc />
    public override object? Config => DriverConfig;

    /// <summary>Corners of the current result, for the selector.</summary>
    public ObservableCollection<CornerChoice> Corners { get; } = new();

    /// <summary>Which figure is shown.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsSummary))]
    [NotifyPropertyChangedFor(nameof(IsDetail))]
    [NotifyPropertyChangedFor(nameof(IsMap))]
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

    /// <summary>Toggle binding for the map.</summary>
    public bool IsMap
    {
        get => ViewMode == DriverViewMode.Map;
        set => ViewMode = value ? DriverViewMode.Map : DriverViewMode.Summary;
    }

    /// <summary>The corner shown in the detail view.</summary>
    [ObservableProperty]
    private CornerChoice? _selectedCorner;

    partial void OnSelectedCornerChanged(CornerChoice? value)
    {
        if (value is not null && ViewMode != DriverViewMode.Detail && !_keepViewMode)
        {
            ViewMode = DriverViewMode.Detail; // selecting a corner switches to the detail view, as in Python
        }

        UpdateDetail();
    }

    partial void OnViewModeChanged(DriverViewMode value) => UpdateDetail();

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

    /// <summary>Whether the stats pane is shown.</summary>
    [ObservableProperty]
    private bool _showStats;

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        if (_a.Log is not null)
        {
            DriverConfig.UpdateAvailableChannels("A", _a.Log.ChannelNames.ToList());
            if (_a.Profile is { } pa)
            {
                DriverConfig.SetFromProfile(pa, "A");
            }

            AutoThrottleThreshold(_a, DriverConfig.ThresholdsA, "A");
            _lastThrottleChannel = DriverConfig.GetChannelNames("A")["throttle"];
        }

        if (_b.Log is not null)
        {
            DriverConfig.HasSessionB = true;
            DriverConfig.UpdateAvailableChannels("B", _b.Log.ChannelNames.ToList());
            if (_b.Profile is { } pb && (_b.LoggerId, _b.VehicleName) != (_a.LoggerId, _a.VehicleName))
            {
                DriverConfig.SyncWithA = false;
                DriverConfig.SetFromProfile(pb, "B");
            }

            AutoThrottleThreshold(_b, DriverConfig.ThresholdsB, "B");
        }

        _resultA = null;
        _resultB = null;
        base.OnSessionLoaded();
    }

    private void OnConfigChanged()
    {
        // A throttle channel change re-derives the auto threshold, as in Python.
        var current = DriverConfig.GetChannelNames("A")["throttle"];
        if (_lastThrottleChannel is not null && current != _lastThrottleChannel && AutoThrottleThreshold(_a, DriverConfig.ThresholdsA, "A"))
        {
            _lastThrottleChannel = current;
        }

        OnSelectionChanged();
    }

    /// <summary>Throttle threshold = 95 % of the channel's 95th percentile, rounded to 0.1 (<c>_auto_set_throttle_threshold</c>).</summary>
    private bool AutoThrottleThreshold(SessionViewModel session, DriverThresholdsViewModel thresholds, string which)
    {
        if (session.Log is null || !session.Log.TryGetChannel(DriverConfig.GetChannelNames(which)["throttle"], out var channel) || channel.Length == 0)
        {
            return false;
        }

        var peak = Numerics.Percentile(channel.Values.Where(v => !double.IsNaN(v)).ToArray(), 95);
        if (peak <= 0)
        {
            return false;
        }

        thresholds.SetThrottleSilently(Math.Round(peak * 0.95, 1, MidpointRounding.ToEven));
        if (which == "A" && DriverConfig.SyncWithA)
        {
            DriverConfig.ThresholdsB.CopyFrom(DriverConfig.ThresholdsA);
        }

        return true;
    }

    private sealed record Parameters(LogFile A, List<int> LapsA, Dictionary<string, string> NamesA, double CornerA, double ThrottleA, double SustainA, LogFile? B, List<int> LapsB, Dictionary<string, string> NamesB, double CornerB, double ThrottleB, double SustainB);

    /// <inheritdoc />
    protected override object? GetParameters()
    {
        if (_a.Log is null)
        {
            return null;
        }

        var lapsA = _a.SelectedLaps;
        if (lapsA.Count == 0)
        {
            Status = "Select laps to analyze";
            return null;
        }

        var lapsB = new List<int>();
        if (_b.Log is not null)
        {
            lapsB = _b.SelectedLaps;
            if (lapsB.Count == 0)
            {
                Status = "Select laps from Session B";
                return null;
            }
        }

        var ta = DriverConfig.GetThresholds("A");
        var tb = DriverConfig.GetThresholds("B");
        return new Parameters(_a.Log, lapsA, DriverConfig.GetChannelNames("A"), ta.CornerThresholdValue, ta.ThrottleThresholdValue, ta.SustainMsValue,
            _b.Log, lapsB, DriverConfig.GetChannelNames("B"), tb.CornerThresholdValue, tb.ThrottleThresholdValue, tb.SustainMsValue);
    }

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        _resultA = DriverConsistency.Analyze(p.A, p.LapsA, p.NamesA, p.CornerA, p.ThrottleA, p.SustainA);
        _resultB = p.B is not null && p.LapsB.Count > 0 ? DriverConsistency.Analyze(p.B, p.LapsB, p.NamesB, p.CornerB, p.ThrottleB, p.SustainB) : null;
    }

    /// <inheritdoc />
    protected override void Display()
    {
        if (_resultA is null)
        {
            return;
        }

        var previous = SelectedCorner?.Index ?? 0;
        Corners.Clear();
        for (var i = 0; i < _resultA.Corners.Count; i++)
        {
            Corners.Add(new CornerChoice(i, $"{_resultA.Corners[i].Name} ({_resultA.Corners[i].Direction})"));
        }

        SummarySpec = DriverChartBuilder.Summary(_resultA, _resultB, _a.SessionLabel, _b.SessionLabel);
        MapSpec = DriverChartBuilder.Map(_resultA);
        Stats = DriverChartBuilder.Tables(_resultA, _resultB);
        var keep = Corners.Count > 0 ? Corners[Math.Min(previous, Corners.Count - 1)] : null;
        if (!Equals(SelectedCorner, keep))
        {
            // Re-selecting after a rerun keeps the current view mode.
            _keepViewMode = true;
            SelectedCorner = keep;
            _keepViewMode = false;
        }

        UpdateDetail();

        var totalTa = _resultA.CornerData.Sum(cd => cd.TaValues.Count);
        var threshold = DriverConfig.GetThresholds("A").ThrottleThreshold;
        Status = totalTa > 0
            ? $"Done - {_resultA.Corners.Count} corners, {totalTa} TA pts (thr: {threshold})"
            : $"Done - {_resultA.Corners.Count} corners, 0 TA - {DiagnoseEmptyTa(threshold)}";
    }

    private string DiagnoseEmptyTa(string threshold)
    {
        var log = _a.Log;
        if (log is null)
        {
            return "try lower threshold";
        }

        var lateralName = DriverConfig.GetChannelNames("A")["lateral_g"];
        if (!log.TryGetChannel(lateralName, out var lateral))
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
        if (_resultA is null || SelectedCorner is null || SelectedCorner.Index >= _resultA.CornerData.Count)
        {
            DetailSpec = null;
            return;
        }

        DetailSpec = DriverChartBuilder.Detail(_resultA.CornerData[SelectedCorner.Index]);
    }
}
