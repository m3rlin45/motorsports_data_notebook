using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Front/rear track widths as validated text fields (empty until a profile or the user supplies them).</summary>
public sealed partial class TrackWidthsViewModel : ObservableObject
{
    /// <summary>Create empty fields; <paramref name="onChanged"/> is called when the user edits one.</summary>
    public TrackWidthsViewModel(Action onChanged)
    {
        Front = new NumericFieldViewModel("Front track width", "0.#", min: 0, minExclusive: true, allowEmpty: true, onChanged: onChanged);
        Rear = new NumericFieldViewModel("Rear track width", "0.#", min: 0, minExclusive: true, allowEmpty: true, onChanged: onChanged);
    }

    /// <summary>Front track width, mm.</summary>
    public NumericFieldViewModel Front { get; }

    /// <summary>Rear track width, mm.</summary>
    public NumericFieldViewModel Rear { get; }

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    /// <summary>Fill silently (empty when null).</summary>
    public void Set(TrackWidths? widths)
    {
        Front.Set(widths?.FrontMm);
        Rear.Set(widths?.RearMm);
    }

    /// <summary>Copy another pair's text silently.</summary>
    public void CopyFrom(TrackWidthsViewModel other)
    {
        Front.CopyFrom(other.Front);
        Rear.CopyFrom(other.Rear);
    }

    /// <summary>The widths, or null while a field is empty or invalid.</summary>
    public TrackWidths? ToTrackWidths() =>
        Front.Value is { } f && Rear.Value is { } r ? new TrackWidths(f, r) : null;
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
        WidthsA = new TrackWidthsViewModel(() => OnWidthsChanged(SessionSlot.A));
        WidthsB = new TrackWidthsViewModel(() => OnWidthsChanged(SessionSlot.B)) { IsEditable = false };
    }

    /// <summary>Session A track widths (mm).</summary>
    public TrackWidthsViewModel WidthsA { get; }

    /// <summary>Session B track widths (mm).</summary>
    public TrackWidthsViewModel WidthsB { get; }

    /// <summary>The widths being shown/edited.</summary>
    public TrackWidthsViewModel Widths => ActiveSession == SessionSlot.B ? WidthsB : WidthsA;

    /// <summary>Raised when the widths change.</summary>
    public event Action? WidthsChanged;

    /// <summary>Widths for a session (B follows A while synced).</summary>
    public TrackWidths? GetTrackWidths(SessionSlot session) => (session == SessionSlot.B && !SyncWithA ? WidthsB : WidthsA).ToTrackWidths();

    private void OnWidthsChanged(SessionSlot session)
    {
        if (session == SessionSlot.A && SyncWithA)
        {
            WidthsB.CopyFrom(WidthsA);
        }

        WidthsChanged?.Invoke();
    }

    /// <inheritdoc />
    protected override IEnumerable<NumericFieldViewModel> NumericFields(SessionSlot session)
    {
        var widths = session == SessionSlot.B ? WidthsB : WidthsA;
        return new[] { widths.Front, widths.Rear };
    }

    /// <inheritdoc />
    protected override void CopyExtrasAToB() => WidthsB.CopyFrom(WidthsA);

    /// <inheritdoc />
    protected override void OnSyncChanged(bool synced) => WidthsB.IsEditable = !synced;

    /// <inheritdoc />
    protected override void SetExtrasFromProfile(VehicleProfile profile, SessionSlot session) =>
        (session == SessionSlot.B ? WidthsB : WidthsA).Set(profile.TrackWidths);

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
    private const string MapTitle = "Track map — hover near the line to highlight its samples";
    private static readonly string[] Colors = { ChartColors.SessionA, ChartColors.SessionB };
    private static readonly string[] Highlights = { ChartColors.HighlightA, ChartColors.HighlightB };

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
            // Binned means over a single cloud are ink so they stand out from it; in a comparison they take their session's colour.
            var binColor = single ? ChartColors.Ink : color;
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
        return new ScatterGridSpec(title, paths, panels, 2, MapTitle);
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
public sealed partial class RollGradientViewModel : AnalysisTabViewModel<RollGradientViewModel.Parameters, RollGradientViewModel.Result>
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private readonly SuspensionConfigViewModel _suspension;

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
        MinLateralG = new NumericFieldViewModel("Min |G| for fit", "0.0#", min: 0, onChanged: OnSelectionChanged);
        MinLateralG.Set(0);
    }

    /// <inheritdoc />
    public override string Header => "Roll Gradient";

    /// <summary>Channels and track widths.</summary>
    public RollGradientConfigViewModel RollConfig { get; }

    /// <inheritdoc />
    public override object? Config => RollConfig;

    /// <inheritdoc />
    public override bool HasStats => true;

    /// <summary>Minimum |lateral G| for the fits.</summary>
    public NumericFieldViewModel MinLateralG { get; }

    /// <summary>The figure.</summary>
    [ObservableProperty]
    private ScatterGridSpec? _spec;

    /// <summary>Stats tables.</summary>
    [ObservableProperty]
    private IReadOnlyList<StatsTable> _stats = Array.Empty<StatsTable>();

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        RollConfig.OnSessionsChanged(_a, _b);
        base.OnSessionLoaded();
    }

    /// <summary>Everything one analysis reads, snapshotted on the UI thread.</summary>
    public sealed record Parameters(
        LogFile A,
        List<int> LapsA,
        Dictionary<string, string> NamesA,
        MotionRatios RatiosA,
        TrackWidths WidthsA,
        string LabelA,
        LogFile? B,
        List<int> LapsB,
        Dictionary<string, string> NamesB,
        MotionRatios RatiosB,
        TrackWidths WidthsB,
        string LabelB,
        double MinLateralG);

    /// <summary>The roll gradients of Session A and, in a comparison, Session B.</summary>
    public sealed record Result(RollGradientResult A, RollGradientResult? B);

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
        var logB = lapsB.Count > 0 ? _b.Log : null;
        var error = RollConfig.FirstError(SessionSlot.A)
            ?? _suspension.FirstError(SessionSlot.A)
            ?? (logB is null ? null : RollConfig.FirstError(SessionSlot.B) ?? _suspension.FirstError(SessionSlot.B))
            ?? MinLateralG.Error;
        if (error is not null)
        {
            return Hint(error);
        }

        var widthsA = RollConfig.GetTrackWidths(SessionSlot.A);
        if (widthsA is null)
        {
            return Hint("Enter the front and rear track widths (mm) under Channels");
        }

        var ratiosA = _suspension.GetMotionRatios(SessionSlot.A)!;
        return new Parameters(
            _a.Log,
            lapsA,
            RollConfig.GetChannelNames(SessionSlot.A),
            ratiosA,
            widthsA,
            _a.SessionLabel,
            logB,
            lapsB,
            RollConfig.GetChannelNames(SessionSlot.B),
            _suspension.GetMotionRatios(SessionSlot.B) ?? ratiosA,
            RollConfig.GetTrackWidths(SessionSlot.B) ?? widthsA,
            _b.SessionLabel,
            MinLateralG.Value ?? 0);
    }

    /// <inheritdoc />
    protected override Result Analyze(Parameters p, CancellationToken cancellation)
    {
        var a = RollGradient.AnalyzeRollGradientMultiLap(p.A, p.LapsA, p.NamesA, p.RatiosA, p.WidthsA, minLateralG: p.MinLateralG, cancellation: cancellation);
        var b = p.B is null ? null : RollGradient.AnalyzeRollGradientMultiLap(p.B, p.LapsB, p.NamesB, p.RatiosB, p.WidthsB, minLateralG: p.MinLateralG, cancellation: cancellation);
        return new Result(a, b);
    }

    /// <inheritdoc />
    protected override void Display(Parameters p, Result result)
    {
        Spec = RollGradientChartBuilder.Build(result.A, result.B, p.LabelA, p.LabelB);
        var tables = new List<StatsTable> { RollGradientChartBuilder.Table(result.A, p.LabelA) };
        if (result.B is not null)
        {
            tables.Add(RollGradientChartBuilder.Table(result.B, p.LabelB));
        }

        Stats = tables;
    }

    /// <inheritdoc />
    protected override void ClearDisplay()
    {
        Spec = null;
        Stats = Array.Empty<StatsTable>();
    }

    /// <inheritdoc />
    public override void SaveState(IDictionary<string, string> state)
    {
        base.SaveState(state);
        if (MinLateralG.IsValid)
        {
            state["min_g"] = MinLateralG.Text;
        }
    }

    /// <inheritdoc />
    public override void RestoreState(IReadOnlyDictionary<string, string> state)
    {
        base.RestoreState(state);
        if (state.TryGetValue("min_g", out var minG) && double.TryParse(minG, NumberStyles.Float, CultureInfo.InvariantCulture, out var g) && g >= 0)
        {
            MinLateralG.Set(g);
        }
    }
}
