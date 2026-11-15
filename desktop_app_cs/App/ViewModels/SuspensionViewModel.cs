using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Four pot/wheel motion ratios as validated text fields.</summary>
public sealed partial class MotionRatiosViewModel : ObservableObject
{
    /// <summary>Create with the Toyota 86 defaults; <paramref name="onChanged"/> is called when the user edits a field.</summary>
    public MotionRatiosViewModel(Action onChanged)
    {
        NumericFieldViewModel Field(string corner) => new($"{corner} motion ratio", "0.###", min: 0, minExclusive: true, onChanged: onChanged);
        FrontLeft = Field("Front-left");
        FrontRight = Field("Front-right");
        RearLeft = Field("Rear-left");
        RearRight = Field("Rear-right");
        Set(MotionRatios.Toyota86Zn6);
    }

    /// <summary>Front-left ratio.</summary>
    public NumericFieldViewModel FrontLeft { get; }

    /// <summary>Front-right ratio.</summary>
    public NumericFieldViewModel FrontRight { get; }

    /// <summary>Rear-left ratio.</summary>
    public NumericFieldViewModel RearLeft { get; }

    /// <summary>Rear-right ratio.</summary>
    public NumericFieldViewModel RearRight { get; }

    /// <summary>The four fields, FL, FR, RL, RR.</summary>
    public IEnumerable<NumericFieldViewModel> Fields => new[] { FrontLeft, FrontRight, RearLeft, RearRight };

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    /// <summary>Fill silently from <paramref name="ratios"/>.</summary>
    public void Set(MotionRatios ratios)
    {
        FrontLeft.Set(ratios.FrontLeft);
        FrontRight.Set(ratios.FrontRight);
        RearLeft.Set(ratios.RearLeft);
        RearRight.Set(ratios.RearRight);
    }

    /// <summary>Copy another set's text silently.</summary>
    public void CopyFrom(MotionRatiosViewModel other)
    {
        FrontLeft.CopyFrom(other.FrontLeft);
        FrontRight.CopyFrom(other.FrontRight);
        RearLeft.CopyFrom(other.RearLeft);
        RearRight.CopyFrom(other.RearRight);
    }

    /// <summary>The ratios, or null while a field does not hold a positive number.</summary>
    public MotionRatios? ToMotionRatios() =>
        FrontLeft.Value is { } fl && FrontRight.Value is { } fr && RearLeft.Value is { } rl && RearRight.Value is { } rr
            ? new MotionRatios(fl, fr, rl, rr)
            : null;
}

/// <summary>Suspension tab config: the four shock channels plus motion ratios, per session.</summary>
public sealed class SuspensionConfigViewModel : ChannelConfigViewModel
{
    /// <summary>Create the config.</summary>
    public SuspensionConfigViewModel()
        : base(new[] { ("shock_fl", "FL Shock"), ("shock_fr", "FR Shock"), ("shock_rl", "RL Shock"), ("shock_rr", "RR Shock") })
    {
        RatiosA = new MotionRatiosViewModel(() => OnRatiosChanged(SessionSlot.A));
        RatiosB = new MotionRatiosViewModel(() => OnRatiosChanged(SessionSlot.B)) { IsEditable = false };
    }

    /// <summary>Session A motion ratios.</summary>
    public MotionRatiosViewModel RatiosA { get; }

    /// <summary>Session B motion ratios.</summary>
    public MotionRatiosViewModel RatiosB { get; }

    /// <summary>The ratios being shown/edited.</summary>
    public MotionRatiosViewModel Ratios => ActiveSession == SessionSlot.B ? RatiosB : RatiosA;

    /// <summary>Raised when a ratio value changes (not when the A/B switch only changes which set is shown).</summary>
    public event Action? RatiosChanged;

    /// <summary>Ratios for a session (B follows A while synced); null while a field is invalid.</summary>
    public MotionRatios? GetMotionRatios(SessionSlot session) =>
        (session == SessionSlot.B && !SyncWithA ? RatiosB : RatiosA).ToMotionRatios();

    private void OnRatiosChanged(SessionSlot session)
    {
        if (session == SessionSlot.A && SyncWithA)
        {
            RatiosB.CopyFrom(RatiosA);
        }

        RatiosChanged?.Invoke();
    }

    /// <inheritdoc />
    protected override IEnumerable<NumericFieldViewModel> NumericFields(SessionSlot session) =>
        (session == SessionSlot.B ? RatiosB : RatiosA).Fields;

    /// <inheritdoc />
    protected override void CopyExtrasAToB() => RatiosB.CopyFrom(RatiosA);

    /// <inheritdoc />
    protected override void OnSyncChanged(bool synced) => RatiosB.IsEditable = !synced;

    /// <inheritdoc />
    protected override void SetExtrasFromProfile(VehicleProfile profile, SessionSlot session) =>
        (session == SessionSlot.B ? RatiosB : RatiosA).Set(profile.MotionRatios);

    /// <inheritdoc />
    protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.PropertyName == nameof(ActiveSession))
        {
            OnPropertyChanged(nameof(Ratios));
        }
    }
}

/// <summary>A small table for the stats pane.</summary>
public sealed record StatsTable(string Title, IReadOnlyList<string> Columns, IReadOnlyList<IReadOnlyList<string>> Rows)
{
    /// <summary>The table as tab-separated text (title, header row, rows), ready to paste into a spreadsheet.</summary>
    public string ToTsv() =>
        string.Join("\n", new[] { Title, string.Join("\t", Columns) }.Concat(Rows.Select(r => string.Join("\t", r))));
}

/// <summary>Builds the chart spec and stats tables from suspension results.</summary>
public static class SuspensionChartBuilder
{
    /// <summary>The hover label: which bar, its velocity bin and the share of the lap spent there.</summary>
    private const string HoverFormat = "{0}\nVelocity: {1:0} mm/s\nTime: {2:0.00}%";

    /// <summary>
    /// The 2×2 histogram grid for one session, or the side-by-side comparison of two. With
    /// <paramref name="scope"/> the bars are a region's and it says what they cover
    /// ("Turn 3 · 41 s"); <paramref name="wholeA"/> is then the whole laps' distribution, outlined
    /// over the bars for reference. A comparison has no such reference: its two sessions are
    /// compared with each other, not with their own whole laps.
    /// </summary>
    public static HistogramGridSpec Build(
        VelocityHistogramResult a,
        VelocityHistogramResult? b,
        string labelA,
        string labelB,
        VelocityHistogramResult? wholeA = null,
        string scope = "")
    {
        var r = a.VelocityRanges;
        var bands = new List<BandSpec>
        {
            new(-r.Friction, r.Friction, ChartColors.BandFriction, 0.15, "Friction"),
            new(r.Friction, r.Slow, ChartColors.BandSlow),
            new(-r.Slow, -r.Friction, ChartColors.BandSlow),
            new(r.Slow, r.Fast, ChartColors.BandFast),
            new(-r.Fast, -r.Slow, ChartColors.BandFast),
            new(r.Fast, 300, ChartColors.BandCurb),
            new(-300, -r.Fast, ChartColors.BandCurb),
        };

        var titles = new[] { "Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)" };
        var cornersA = a.Corners.ToList();
        var cornersB = b?.Corners.ToList();
        var wholeCornersA = b is null ? wholeA?.Corners.ToList() : null;
        var panels = new List<HistogramPanelSpec>();
        for (var i = 0; i < 4; i++)
        {
            var ca = cornersA[i];
            var binWidth = ca.BinEdges.Length > 1 ? ca.BinEdges[1] - ca.BinEdges[0] : 10;
            if (cornersB is null)
            {
                var colors = ca.BinCenters.Select(c => c >= 0 ? ChartColors.Bump : ChartColors.Rebound).ToArray();
                var annotation = string.Format(CultureInfo.InvariantCulture, "Skew: {0:0.00}\nStd: {1:0} mm/s", ca.Skew, ca.Std);
                var series = new List<BarSeriesSpec> { new(wholeCornersA is null ? string.Empty : "Selected regions", ca.BinCenters, ca.Histogram, BarColors: colors) };
                if (wholeCornersA is not null)
                {
                    series.Add(new BarSeriesSpec("Whole lap", wholeCornersA[i].BinCenters, wholeCornersA[i].Histogram, ChartColors.Ink, Outline: true));
                }

                panels.Add(new HistogramPanelSpec(titles[i], series, binWidth, annotation));
            }
            else
            {
                var cb = cornersB[i];
                panels.Add(new HistogramPanelSpec(titles[i], new[]
                {
                    new BarSeriesSpec(labelA, ca.BinCenters, ca.Histogram, ChartColors.SessionA),
                    new BarSeriesSpec(labelB, cb.BinCenters, cb.Histogram, ChartColors.SessionB),
                }, binWidth));
            }
        }

        var title = b is null ? "Suspension Velocity Distribution" : $"Comparison: {labelA} vs {labelB}";
        if (scope.Length > 0)
        {
            title += $" · {scope}";
        }

        return new HistogramGridSpec(title, panels, 2, new XRange(-300, 300), "Velocity (mm/s)", "Time (%)", bands, HoverFormat);
    }

    /// <summary>The three stats tables of the Python stats panel for one session.</summary>
    public static List<StatsTable> StatsTables(VelocityHistogramResult result, string label)
    {
        static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);
        var corners = result.Corners.ToList();

        var stats = new StatsTable(
            $"Statistics — {label}",
            new[] { "Corner", "Skew", "Kurtosis", "Mean (mm/s)", "Std (mm/s)", "Zero Bin %", "Friction %", "Slow Bump %", "Slow Rebound %", "Fast Bump %", "Fast Rebound %", "Curb %" },
            corners.Select(c => (IReadOnlyList<string>)new[] { c.CornerName, F(c.Skew, "0.00"), F(c.Kurtosis, "0.00"), F(c.Mean, "0.0"), F(c.Std, "0.0"), F(c.PctZeroBin, "0.0"), F(c.PctFriction, "0.0"), F(c.PctSlowBump, "0.0"), F(c.PctSlowRebound, "0.0"), F(c.PctFastBump, "0.0"), F(c.PctFastRebound, "0.0"), F(c.PctCurb, "0.0") }).ToList());

        var symmetry = new StatsTable(
            "Bump vs Rebound",
            new[] { "Corner", "Total Bump %", "Total Rebound %", "Bump/Total %", "Slow Bump %", "Slow Rebound %", "Fast Bump %", "Fast Rebound %" },
            corners.Select(c => (IReadOnlyList<string>)new[] { c.CornerName, F(c.PctSlowBump + c.PctFastBump, "0.0"), F(c.PctSlowRebound + c.PctFastRebound, "0.0"), F(c.BumpOfTotalPct, "0.0"), F(c.PctSlowBump, "0.0"), F(c.PctSlowRebound, "0.0"), F(c.PctFastBump, "0.0"), F(c.PctFastRebound, "0.0") }).ToList());

        var (lr, fr) = result.GroupDifferences();
        var comparison = new StatsTable(
            "Left vs Right, Front vs Rear (percentage-point differences)",
            new[] { "Comparison", "Zero Bin", "Friction", "Slow Bump", "Slow Rebound", "Fast Bump", "Fast Rebound", "Curb" },
            new IReadOnlyList<string>[]
            {
                new[] { "Left vs Right", F(lr.ZeroBin, "+0.0;-0.0"), F(lr.Friction, "+0.0;-0.0"), F(lr.SlowBump, "+0.0;-0.0"), F(lr.SlowRebound, "+0.0;-0.0"), F(lr.FastBump, "+0.0;-0.0"), F(lr.FastRebound, "+0.0;-0.0"), F(lr.Curb, "+0.0;-0.0") },
                new[] { "Front vs Rear", F(fr.ZeroBin, "+0.0;-0.0"), F(fr.Friction, "+0.0;-0.0"), F(fr.SlowBump, "+0.0;-0.0"), F(fr.SlowRebound, "+0.0;-0.0"), F(fr.FastBump, "+0.0;-0.0"), F(fr.FastRebound, "+0.0;-0.0"), F(fr.Curb, "+0.0;-0.0") },
            });

        return new List<StatsTable> { stats, symmetry, comparison };
    }
}

/// <summary>The Suspension tab: velocity histograms for Session A, or A vs B.</summary>
public sealed partial class SuspensionViewModel : AnalysisTabViewModel<SuspensionViewModel.Parameters, SuspensionViewModel.Result>
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;

    /// <summary>Create the tab over the two session slots.</summary>
    public SuspensionViewModel(SessionViewModel a, SessionViewModel b)
    {
        _a = a;
        _b = b;
        SuspensionConfig = new SuspensionConfigViewModel();
        SuspensionConfig.Changed += OnSelectionChanged;
        SuspensionConfig.RatiosChanged += OnSelectionChanged;
    }

    /// <inheritdoc />
    public override string Header => "Suspension";

    /// <summary>Channels and motion ratios.</summary>
    public SuspensionConfigViewModel SuspensionConfig { get; }

    /// <inheritdoc />
    public override object? Config => SuspensionConfig;

    /// <inheritdoc />
    public override bool HasStats => true;

    /// <summary>The chart to draw.</summary>
    [ObservableProperty]
    private HistogramGridSpec? _spec;

    /// <summary>Stats tables for the stats pane (A, then B when present).</summary>
    [ObservableProperty]
    private IReadOnlyList<StatsTable> _stats = Array.Empty<StatsTable>();

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        SuspensionConfig.OnSessionsChanged(_a, _b);
        base.OnSessionLoaded();
    }

    /// <summary>Everything one analysis reads, snapshotted on the UI thread.</summary>
    public sealed record Parameters(
        LogFile A,
        List<int> LapsA,
        Dictionary<string, string> NamesA,
        MotionRatios RatiosA,
        string LabelA,
        LogFile? B,
        List<int> LapsB,
        Dictionary<string, string> NamesB,
        MotionRatios RatiosB,
        string LabelB,
        RegionSelection Regions);

    /// <summary>
    /// The histograms of Session A and, in a comparison, Session B, over the ticked regions when
    /// there are any (<paramref name="RegionSecondsA"/> is how much of Session A's laps they cover).
    /// <paramref name="WholeA"/> is then Session A's whole-lap result for reference: a single
    /// session only, and only when the reference is wanted.
    /// </summary>
    public sealed record Result(
        VelocityHistogramResult A,
        VelocityHistogramResult? B,
        VelocityHistogramResult? WholeA = null,
        double RegionSecondsA = 0);

    /// <inheritdoc />
    public override RegionUse RegionUse => RegionUse.Filter;

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
        if (SuspensionConfig.FirstError(SessionSlot.A) is { } errorA)
        {
            return Hint(errorA);
        }

        if (logB is not null && SuspensionConfig.FirstError(SessionSlot.B) is { } errorB)
        {
            return Hint(errorB);
        }

        var ratiosA = SuspensionConfig.GetMotionRatios(SessionSlot.A)!;
        return new Parameters(
            _a.Log,
            lapsA,
            SuspensionConfig.GetChannelNames(SessionSlot.A),
            ratiosA,
            _a.SessionLabel,
            logB,
            lapsB,
            SuspensionConfig.GetChannelNames(SessionSlot.B),
            SuspensionConfig.GetMotionRatios(SessionSlot.B) ?? ratiosA,
            _b.SessionLabel,
            RegionSnapshot());
    }

    /// <inheritdoc />
    protected override Result Analyze(Parameters p, CancellationToken cancellation)
    {
        var windowsA = p.Regions.Windows(p.A, p.LapsA, SessionSlot.A);
        var a = Suspension.AnalyzeSuspensionVelocityMultiLap(p.A, p.LapsA, p.NamesA, p.RatiosA, cancellation: cancellation, windowsByLap: windowsA);
        var b = p.B is null
            ? null
            : Suspension.AnalyzeSuspensionVelocityMultiLap(p.B, p.LapsB, p.NamesB, p.RatiosB, cancellation: cancellation, windowsByLap: p.Regions.Windows(p.B, p.LapsB, SessionSlot.B));

        // The whole laps behind a region: for one session only. Two sessions are compared with each other.
        var whole = p.B is null && p.Regions is { IsWholeLap: false, ShowWholeLap: true }
            ? Suspension.AnalyzeSuspensionVelocityMultiLap(p.A, p.LapsA, p.NamesA, p.RatiosA, cancellation: cancellation)
            : null;
        return new Result(a, b, whole, windowsA?.Values.Sum(TrackRegions.DurationS) ?? 0);
    }

    /// <inheritdoc />
    protected override void Display(Parameters p, Result result)
    {
        // Restricted to ticked regions, the bars and the tables are the regions'.
        var inRegions = !p.Regions.IsWholeLap;
        var scope = inRegions ? $"{p.Regions.Label} · {result.RegionSecondsA.ToString("0", CultureInfo.InvariantCulture)} s" : string.Empty;
        string Named(string label) => inRegions ? $"{label}, {p.Regions.Label}" : label;
        Spec = SuspensionChartBuilder.Build(result.A, result.B, p.LabelA, p.LabelB, result.WholeA, scope);
        var tables = SuspensionChartBuilder.StatsTables(result.A, Named(p.LabelA));
        if (result.B is not null)
        {
            tables.AddRange(SuspensionChartBuilder.StatsTables(result.B, Named(p.LabelB)));
        }

        Stats = tables;
        if (inRegions && result.RegionSecondsA < LowDataSeconds)
        {
            SetStatus($"Only {result.RegionSecondsA.ToString("0.#", CultureInfo.InvariantCulture)} s of data in {p.Regions.Label}: tick more laps or regions for a steadier histogram", StatusKind.Hint);
        }
    }

    /// <summary>Below this much time inside the ticked regions the histogram is flagged as thin.</summary>
    public const double LowDataSeconds = 10;

    /// <inheritdoc />
    protected override void ClearDisplay()
    {
        Spec = null;
        Stats = Array.Empty<StatsTable>();
    }
}
