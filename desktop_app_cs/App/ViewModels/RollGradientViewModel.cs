using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Front/rear track widths as editable text.</summary>
public sealed partial class TrackWidthsViewModel : ObservableObject
{
    private readonly Action _onChanged;
    private bool _suppress;

    /// <summary>Create empty fields.</summary>
    public TrackWidthsViewModel(Action onChanged)
    {
        _onChanged = onChanged;
    }

    [ObservableProperty]
    private string _front = string.Empty;

    [ObservableProperty]
    private string _rear = string.Empty;

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    partial void OnFrontChanged(string value) => Raise();
    partial void OnRearChanged(string value) => Raise();

    private void Raise()
    {
        if (!_suppress)
        {
            _onChanged();
        }
    }

    /// <summary>Fill silently (empty when null).</summary>
    public void Set(TrackWidths? widths)
    {
        _suppress = true;
        Front = widths is null ? string.Empty : widths.FrontMm.ToString("0.#", CultureInfo.InvariantCulture);
        Rear = widths is null ? string.Empty : widths.RearMm.ToString("0.#", CultureInfo.InvariantCulture);
        _suppress = false;
    }

    /// <summary>The widths, or null when a field is empty or not positive.</summary>
    public TrackWidths? ToTrackWidths()
    {
        if (double.TryParse(Front, NumberStyles.Float, CultureInfo.InvariantCulture, out var f) && f > 0
            && double.TryParse(Rear, NumberStyles.Float, CultureInfo.InvariantCulture, out var r) && r > 0)
        {
            return new TrackWidths(f, r);
        }

        return null;
    }
}

/// <summary>Roll Gradient tab config: shock, lateral-G and GPS channels plus track widths per session.</summary>
public sealed class RollGradientConfigViewModel : ChannelConfigViewModel
{
    /// <summary>Create the config.</summary>
    public RollGradientConfigViewModel()
        : base(new[]
        {
            ("shock_fl", "FL Shock"), ("shock_fr", "FR Shock"), ("shock_rl", "RL Shock"), ("shock_rr", "RR Shock"),
            ("lateral_g", "Lateral G"), ("gps_latitude", "GPS Lat"), ("gps_longitude", "GPS Lon"),
        })
    {
        WidthsA = new TrackWidthsViewModel(() => OnWidthsChanged("A"));
        WidthsB = new TrackWidthsViewModel(() => OnWidthsChanged("B")) { IsEditable = false };
    }

    /// <summary>Session A track widths (mm).</summary>
    public TrackWidthsViewModel WidthsA { get; }

    /// <summary>Session B track widths (mm).</summary>
    public TrackWidthsViewModel WidthsB { get; }

    /// <summary>The widths being shown/edited.</summary>
    public TrackWidthsViewModel Widths => ActiveSession == "B" ? WidthsB : WidthsA;

    /// <summary>Raised when the widths change.</summary>
    public event Action? WidthsChanged;

    /// <summary>Widths for a session (B follows A while synced).</summary>
    public TrackWidths? GetTrackWidths(string session) => (session == "B" && !SyncWithA ? WidthsB : WidthsA).ToTrackWidths();

    private void OnWidthsChanged(string session)
    {
        if (session == "A" && SyncWithA)
        {
            WidthsB.Set(WidthsA.ToTrackWidths());
        }

        WidthsChanged?.Invoke();
    }

    /// <inheritdoc />
    protected override void CopyExtrasAToB() => WidthsB.Set(WidthsA.ToTrackWidths());

    /// <inheritdoc />
    protected override void OnSyncChanged(bool synced) => WidthsB.IsEditable = !synced;

    /// <inheritdoc />
    protected override void SetExtrasFromProfile(VehicleProfile profile, string session) =>
        (session == "B" ? WidthsB : WidthsA).Set(profile.TrackWidths);

    /// <inheritdoc />
    protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.PropertyName == nameof(ActiveSession))
        {
            OnPropertyChanged(nameof(Widths));
        }
    }
}

/// <summary>Builds the roll-gradient figure (map + three scatters) and its table.</summary>
public static class RollGradientChartBuilder
{
    private const int MaxScatterPoints = 12000;
    private static readonly string[] Colors = { "#4682b4", "#ff8c00" };
    private static readonly string[] Highlights = { "#ffe066", "#ff6ec7" };

    private sealed record Series(string Label, double[] Lat, double[] RollF, double[] RollR, double[]? X, double[]? Y);

    private static Series Downsample(RollGradientResult r, string label)
    {
        var n = r.Front.LateralG.Length;
        var step = Math.Max(1, n / MaxScatterPoints);
        static double[] Take(double[] a, int step) => step == 1 ? a : Enumerable.Range(0, (a.Length + step - 1) / step).Select(i => a[i * step]).ToArray();
        var hasMap = r.HasTrackMap;
        return new Series(label, Take(r.Front.LateralG, step), Take(r.Front.RollDeg, step), Take(r.Rear.RollDeg, step), hasMap ? Take(r.XM!, step) : null, hasMap ? Take(r.YM!, step) : null);
    }

    private static LineSpec? Fit(double[] xs, double slope, double intercept, string color)
    {
        if (xs.Length == 0 || !double.IsFinite(slope))
        {
            return null;
        }

        var x1 = xs.Min();
        var x2 = xs.Max();
        return new LineSpec(x1, slope * x1 + intercept, x2, slope * x2 + intercept, color);
    }

    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    /// <summary>The figure for one session or an A/B overlay.</summary>
    public static ScatterGridSpec Build(RollGradientResult a, RollGradientResult? b, string labelA, string labelB)
    {
        var results = b is null ? new[] { (a, "Session A") } : new[] { (a, labelA), (b, labelB) };
        var series = results.Select(r => Downsample(r.Item1, r.Item2)).ToList();
        var single = b is null;

        var front = new List<ScatterSeriesSpec>();
        var rear = new List<ScatterSeriesSpec>();
        var ratio = new List<ScatterSeriesSpec>();
        var frontLines = new List<LineSpec>();
        var rearLines = new List<LineSpec>();
        var ratioLines = new List<LineSpec>();
        var frontMarkers = new List<MarkerSeriesSpec>();
        var rearMarkers = new List<MarkerSeriesSpec>();
        var ratioMarkers = new List<MarkerSeriesSpec>();
        var paths = new List<TrackPathSpec>();

        for (var i = 0; i < series.Count; i++)
        {
            var s = series[i];
            var r = results[i].Item1;
            var color = Colors[i % 2];
            var binColor = single ? "#ffffff" : color;
            front.Add(new ScatterSeriesSpec(single ? string.Empty : s.Label, s.Lat, s.RollF, color, i, 0.15));
            rear.Add(new ScatterSeriesSpec(string.Empty, s.Lat, s.RollR, color, i, 0.15));
            ratio.Add(new ScatterSeriesSpec(string.Empty, s.RollR, s.RollF, color, i, 0.15));
            frontMarkers.Add(new MarkerSeriesSpec(r.Front.BinCentersG, r.Front.BinMeanDeg, binColor));
            rearMarkers.Add(new MarkerSeriesSpec(r.Rear.BinCentersG, r.Rear.BinMeanDeg, binColor));
            ratioMarkers.Add(new MarkerSeriesSpec(r.RatioBinCentersDeg, r.RatioBinMeanDeg, binColor));
            if (Fit(s.Lat, r.Front.GradientDegPerG, r.Front.InterceptDeg, color) is { } fl) frontLines.Add(fl);
            if (Fit(s.Lat, r.Rear.GradientDegPerG, r.Rear.InterceptDeg, color) is { } rl) rearLines.Add(rl);
            if (Fit(s.RollR, r.RollRatio, r.RollRatioInterceptDeg, color) is { } tl) ratioLines.Add(tl);
            if (s.X is not null && s.Y is not null)
            {
                paths.Add(new TrackPathSpec(s.Label, s.X, s.Y, color, Highlights[i % 2], i));
            }
        }

        string frontNote, rearNote, ratioNote, warning = string.Empty;
        if (single)
        {
            frontNote = $"gradient = {F(a.Front.GradientDegPerG, "+0.000;-0.000")} deg/g\nR² = {F(a.Front.R2, "0.000")}   n = {a.Front.NSamples}";
            rearNote = $"gradient = {F(a.Rear.GradientDegPerG, "+0.000;-0.000")} deg/g\nR² = {F(a.Rear.R2, "0.000")}   n = {a.Rear.NSamples}";
            ratioNote = $"roll ratio F/R = {F(a.RollRatio, "0.000")}\nR² = {F(a.RollRatioR2, "0.000")}";
            if (!a.SignsConsistent)
            {
                warning = "front/rear roll in opposite directions:\ncheck left/right channel assignment";
            }
        }
        else
        {
            frontNote = $"A {F(a.Front.GradientDegPerG, "+0.000;-0.000")}  B {F(b!.Front.GradientDegPerG, "+0.000;-0.000")} deg/g\nΔ|grad| = {F(b.Front.AbsGradientDegPerG - a.Front.AbsGradientDegPerG, "+0.000;-0.000")}";
            rearNote = $"A {F(a.Rear.GradientDegPerG, "+0.000;-0.000")}  B {F(b.Rear.GradientDegPerG, "+0.000;-0.000")} deg/g\nΔ|grad| = {F(b.Rear.AbsGradientDegPerG - a.Rear.AbsGradientDegPerG, "+0.000;-0.000")}";
            ratioNote = $"roll ratio A {F(a.RollRatio, "0.000")}  B {F(b.RollRatio, "0.000")}\nΔ = {F(b.RollRatio - a.RollRatio, "+0.000;-0.000")}";
        }

        var panels = new[]
        {
            new ScatterPanelSpec("Front roll vs lateral G", "Lateral G (g)", "Front roll (deg)", front, frontLines, frontMarkers, frontNote, ShowLegend: !single),
            new ScatterPanelSpec("Rear roll vs lateral G", "Lateral G (g)", "Rear roll (deg)", rear, rearLines, rearMarkers, rearNote, warning),
            new ScatterPanelSpec("Roll ratio: front roll vs rear roll", "Rear roll (deg)", "Front roll (deg)", ratio, ratioLines, ratioMarkers, ratioNote),
        };

        var title = single
            ? $"Roll Gradient  |  front {F(a.Front.AbsGradientDegPerG, "0.00")} deg/g, rear {F(a.Rear.AbsGradientDegPerG, "0.00")} deg/g, roll ratio {F(a.RollRatio, "0.00")}"
            : $"Roll Gradient: {labelA} vs {labelB}";
        return new ScatterGridSpec(title, paths, panels, 2);
    }

    /// <summary><c>format_roll_gradient_table</c>.</summary>
    public static StatsTable Table(RollGradientResult r, string label)
    {
        IReadOnlyList<string> Row(string axle, double gradient, double intercept, double r2, int n) =>
            new[] { axle, F(gradient, "+0.000;-0.000"), F(Math.Abs(gradient), "0.000"), F(intercept, "+0.000;-0.000"), F(r2, "0.000"), n.ToString(CultureInfo.InvariantCulture) };
        return new StatsTable(
            $"Roll gradient — {label}",
            new[] { "Axle", "Gradient (deg/g)", "|Gradient| (deg/g)", "Intercept (deg)", "R²", "Samples" },
            new[]
            {
                Row("Front", r.Front.GradientDegPerG, r.Front.InterceptDeg, r.Front.R2, r.Front.NSamples),
                Row("Rear", r.Rear.GradientDegPerG, r.Rear.InterceptDeg, r.Rear.R2, r.Rear.NSamples),
                Row("Roll ratio (front/rear)", r.RollRatio, r.RollRatioInterceptDeg, r.RollRatioR2, Math.Min(r.Front.NSamples, r.Rear.NSamples)),
            });
    }
}

/// <summary>The Roll Gradient tab.</summary>
public sealed partial class RollGradientViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private readonly SuspensionConfigViewModel _suspension;
    private RollGradientResult? _resultA;
    private RollGradientResult? _resultB;

    /// <summary>Create the tab; motion ratios come from the Suspension tab's config, as in the Python app.</summary>
    public RollGradientViewModel(SessionViewModel a, SessionViewModel b, SuspensionConfigViewModel suspension)
    {
        _a = a;
        _b = b;
        _suspension = suspension;
        RollConfig = new RollGradientConfigViewModel();
        RollConfig.Changed += OnSelectionChanged;
        RollConfig.WidthsChanged += OnSelectionChanged;
        suspension.RatiosChanged += OnSelectionChanged;
    }

    /// <inheritdoc />
    public override string Header => "Roll Gradient";

    /// <summary>Channels and track widths.</summary>
    public RollGradientConfigViewModel RollConfig { get; }

    /// <inheritdoc />
    public override object? Config => RollConfig;

    /// <summary>Minimum |lateral G| for the fits, as text.</summary>
    [ObservableProperty]
    private string _minLateralG = "0.0";

    partial void OnMinLateralGChanged(string value) => OnSelectionChanged();

    /// <summary>The figure.</summary>
    [ObservableProperty]
    private ScatterGridSpec? _spec;

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
            RollConfig.UpdateAvailableChannels("A", _a.Log.ChannelNames.ToList());
            if (_a.Profile is { } pa)
            {
                RollConfig.SetFromProfile(pa, "A");
            }
        }

        if (_b.Log is not null)
        {
            RollConfig.HasSessionB = true;
            RollConfig.UpdateAvailableChannels("B", _b.Log.ChannelNames.ToList());
            if (_b.Profile is { } pb && (_b.LoggerId, _b.VehicleName) != (_a.LoggerId, _a.VehicleName))
            {
                RollConfig.SyncWithA = false;
                RollConfig.SetFromProfile(pb, "B");
            }
        }

        _resultA = null;
        _resultB = null;
        base.OnSessionLoaded();
    }

    private sealed record Parameters(LogFile A, List<int> LapsA, Dictionary<string, string> NamesA, MotionRatios RatiosA, TrackWidths WidthsA, LogFile? B, List<int> LapsB, Dictionary<string, string> NamesB, MotionRatios RatiosB, TrackWidths WidthsB, double MinLateralG);

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

        var widthsA = RollConfig.GetTrackWidths("A");
        if (widthsA is null)
        {
            Status = "Enter front and rear track widths (mm)";
            return null;
        }

        var lapsB = new List<int>();
        var widthsB = widthsA;
        if (_b.Log is not null)
        {
            lapsB = _b.SelectedLaps;
            if (lapsB.Count == 0)
            {
                Status = "Select laps from Session B";
                return null;
            }

            widthsB = RollConfig.GetTrackWidths("B") ?? widthsA;
        }

        var minG = double.TryParse(MinLateralG, NumberStyles.Float, CultureInfo.InvariantCulture, out var g) ? Math.Max(0, g) : 0;
        return new Parameters(_a.Log, lapsA, RollConfig.GetChannelNames("A"), _suspension.GetMotionRatios("A"), widthsA, _b.Log, lapsB, RollConfig.GetChannelNames("B"), _suspension.GetMotionRatios("B"), widthsB, minG);
    }

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        _resultA = RollGradient.AnalyzeRollGradientMultiLap(p.A, p.LapsA, p.NamesA, p.RatiosA, p.WidthsA, minLateralG: p.MinLateralG);
        _resultB = p.B is not null && p.LapsB.Count > 0 ? RollGradient.AnalyzeRollGradientMultiLap(p.B, p.LapsB, p.NamesB, p.RatiosB, p.WidthsB, minLateralG: p.MinLateralG) : null;
    }

    /// <inheritdoc />
    protected override void Display()
    {
        if (_resultA is null)
        {
            return;
        }

        Spec = RollGradientChartBuilder.Build(_resultA, _resultB, _a.SessionLabel, _b.SessionLabel);
        var tables = new List<StatsTable> { RollGradientChartBuilder.Table(_resultA, _a.SessionLabel) };
        if (_resultB is not null)
        {
            tables.Add(RollGradientChartBuilder.Table(_resultB, _b.SessionLabel));
        }

        Stats = tables;
    }
}
