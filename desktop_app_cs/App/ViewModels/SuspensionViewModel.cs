using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>Four pot/wheel motion ratios as editable text.</summary>
public sealed partial class MotionRatiosViewModel : ObservableObject
{
    private readonly Action _onChanged;
    private bool _suppress;

    /// <summary>Create with the Toyota 86 defaults.</summary>
    public MotionRatiosViewModel(Action onChanged)
    {
        _onChanged = onChanged;
        Set(MotionRatios.Toyota86Zn6);
    }

    [ObservableProperty]
    private string _frontLeft = string.Empty;

    [ObservableProperty]
    private string _frontRight = string.Empty;

    [ObservableProperty]
    private string _rearLeft = string.Empty;

    [ObservableProperty]
    private string _rearRight = string.Empty;

    /// <summary>Whether the fields can be edited.</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    partial void OnFrontLeftChanged(string value) => Raise();
    partial void OnFrontRightChanged(string value) => Raise();
    partial void OnRearLeftChanged(string value) => Raise();
    partial void OnRearRightChanged(string value) => Raise();

    private void Raise()
    {
        if (!_suppress)
        {
            _onChanged();
        }
    }

    /// <summary>Fill silently from <paramref name="ratios"/>.</summary>
    public void Set(MotionRatios ratios)
    {
        _suppress = true;
        FrontLeft = Format(ratios.FrontLeft);
        FrontRight = Format(ratios.FrontRight);
        RearLeft = Format(ratios.RearLeft);
        RearRight = Format(ratios.RearRight);
        _suppress = false;
    }

    private static string Format(double v) => v.ToString("0.###", CultureInfo.InvariantCulture);

    private static double Parse(string text, double fallback) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) && v > 0 ? v : fallback;

    /// <summary>The ratios; unparsable fields fall back to the Toyota 86 values.</summary>
    public MotionRatios ToMotionRatios()
    {
        var d = MotionRatios.Toyota86Zn6;
        return new MotionRatios(Parse(FrontLeft, d.FrontLeft), Parse(FrontRight, d.FrontRight), Parse(RearLeft, d.RearLeft), Parse(RearRight, d.RearRight));
    }
}

/// <summary>Suspension tab config: the four shock channels plus motion ratios, per session.</summary>
public sealed class SuspensionConfigViewModel : ChannelConfigViewModel
{
    /// <summary>Create the config.</summary>
    public SuspensionConfigViewModel()
        : base(new[] { ("shock_fl", "FL Shock"), ("shock_fr", "FR Shock"), ("shock_rl", "RL Shock"), ("shock_rr", "RR Shock") })
    {
        RatiosA = new MotionRatiosViewModel(() => OnRatiosChanged("A"));
        RatiosB = new MotionRatiosViewModel(() => OnRatiosChanged("B")) { IsEditable = false };
    }

    /// <summary>Session A motion ratios.</summary>
    public MotionRatiosViewModel RatiosA { get; }

    /// <summary>Session B motion ratios.</summary>
    public MotionRatiosViewModel RatiosB { get; }

    /// <summary>The ratios being shown/edited.</summary>
    public MotionRatiosViewModel Ratios => ActiveSession == "B" ? RatiosB : RatiosA;

    /// <summary>Ratios for a session (B follows A while synced).</summary>
    public MotionRatios GetMotionRatios(string session) =>
        (session == "B" && !SyncWithA ? RatiosB : RatiosA).ToMotionRatios();

    private void OnRatiosChanged(string session)
    {
        if (session == "A" && SyncWithA)
        {
            RatiosB.Set(RatiosA.ToMotionRatios());
        }

        RaiseChanged();
    }

    private void RaiseChanged() => OnPropertyChanged(nameof(Ratios)); // the base Changed event is raised via entries; ratios notify through the tab

    /// <summary>Raised when ratios change (the base class's Changed covers entries).</summary>
    public event Action? RatiosChanged
    {
        add => _ratiosChanged += value;
        remove => _ratiosChanged -= value;
    }

    private Action? _ratiosChanged;

    /// <inheritdoc />
    protected override void CopyExtrasAToB() => RatiosB.Set(RatiosA.ToMotionRatios());

    /// <inheritdoc />
    protected override void OnSyncChanged(bool synced) => RatiosB.IsEditable = !synced;

    /// <inheritdoc />
    protected override void SetExtrasFromProfile(VehicleProfile profile, string session) =>
        (session == "B" ? RatiosB : RatiosA).Set(profile.MotionRatios);

    /// <inheritdoc />
    protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.PropertyName == nameof(ActiveSession))
        {
            OnPropertyChanged(nameof(Ratios));
        }
        else if (e.PropertyName == nameof(Ratios))
        {
            _ratiosChanged?.Invoke();
        }
    }
}

/// <summary>A small table for the stats pane.</summary>
public sealed record StatsTable(string Title, IReadOnlyList<string> Columns, IReadOnlyList<IReadOnlyList<string>> Rows);

/// <summary>Builds the chart spec and stats tables from suspension results.</summary>
public static class SuspensionChartBuilder
{
    private const string Blue = "#4682b4";
    private const string Red = "#cd5c5c";
    private const string Orange = "#ff8c00";

    /// <summary>The 2×2 histogram grid for one session, or the side-by-side comparison of two.</summary>
    public static HistogramGridSpec Build(VelocityHistogramResult a, VelocityHistogramResult? b, string labelA, string labelB)
    {
        var r = a.VelocityRanges;
        var bands = new List<BandSpec>
        {
            new(-r.Friction, r.Friction, "#808080", 0.15, "Friction"),
            new(r.Friction, r.Slow, "#add8e6"),
            new(-r.Slow, -r.Friction, "#add8e6"),
            new(r.Slow, r.Fast, "#90ee90"),
            new(-r.Fast, -r.Slow, "#90ee90"),
            new(r.Fast, 300, "#f08080"),
            new(-300, -r.Fast, "#f08080"),
        };

        var titles = new[] { "Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)" };
        var cornersA = a.Corners.ToList();
        var cornersB = b?.Corners.ToList();
        var panels = new List<HistogramPanelSpec>();
        for (var i = 0; i < 4; i++)
        {
            var ca = cornersA[i];
            var binWidth = ca.BinEdges.Length > 1 ? ca.BinEdges[1] - ca.BinEdges[0] : 10;
            if (cornersB is null)
            {
                var colors = ca.BinCenters.Select(c => c >= 0 ? Blue : Red).ToArray();
                var annotation = string.Format(CultureInfo.InvariantCulture, "Skew: {0:0.00}\nStd: {1:0} mm/s", ca.Skew, ca.Std);
                panels.Add(new HistogramPanelSpec(titles[i], new[] { new BarSeriesSpec(string.Empty, ca.BinCenters, ca.Histogram, BarColors: colors) }, binWidth, annotation));
            }
            else
            {
                var cb = cornersB[i];
                panels.Add(new HistogramPanelSpec(titles[i], new[]
                {
                    new BarSeriesSpec(labelA, ca.BinCenters, ca.Histogram, Blue),
                    new BarSeriesSpec(labelB, cb.BinCenters, cb.Histogram, Orange),
                }, binWidth));
            }
        }

        var title = b is null ? "Suspension Velocity Distribution" : $"Comparison: {labelA} vs {labelB}";
        return new HistogramGridSpec(title, panels, 2, new XRange(-300, 300), "Velocity (mm/s)", "Time (%)", bands);
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
public sealed partial class SuspensionViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private VelocityHistogramResult? _resultA;
    private VelocityHistogramResult? _resultB;

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

    /// <summary>The chart to draw.</summary>
    [ObservableProperty]
    private HistogramGridSpec? _spec;

    /// <summary>Stats tables for the stats pane (A, then B when present).</summary>
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
            SuspensionConfig.UpdateAvailableChannels("A", _a.Log.ChannelNames.ToList());
            if (_a.Profile is { } pa)
            {
                SuspensionConfig.SetFromProfile(pa, "A");
            }
        }

        if (_b.Log is not null)
        {
            SuspensionConfig.HasSessionB = true;
            SuspensionConfig.UpdateAvailableChannels("B", _b.Log.ChannelNames.ToList());
            if (_b.Profile is { } pb && (_b.LoggerId, _b.VehicleName) != (_a.LoggerId, _a.VehicleName))
            {
                SuspensionConfig.SyncWithA = false;
                SuspensionConfig.SetFromProfile(pb, "B");
            }
        }

        _resultA = null;
        _resultB = null;
        base.OnSessionLoaded();
    }

    private sealed record Parameters(LogFile A, List<int> LapsA, Dictionary<string, string> NamesA, MotionRatios RatiosA, LogFile? B, List<int> LapsB, Dictionary<string, string> NamesB, MotionRatios RatiosB);

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

        return new Parameters(_a.Log, lapsA, SuspensionConfig.GetChannelNames("A"), SuspensionConfig.GetMotionRatios("A"), _b.Log, lapsB, SuspensionConfig.GetChannelNames("B"), SuspensionConfig.GetMotionRatios("B"));
    }

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        _resultA = Suspension.AnalyzeSuspensionVelocityMultiLap(p.A, p.LapsA, p.NamesA, p.RatiosA);
        _resultB = p.B is not null && p.LapsB.Count > 0 ? Suspension.AnalyzeSuspensionVelocityMultiLap(p.B, p.LapsB, p.NamesB, p.RatiosB) : null;
    }

    /// <inheritdoc />
    protected override void Display()
    {
        if (_resultA is null)
        {
            return;
        }

        Spec = SuspensionChartBuilder.Build(_resultA, _resultB, _a.SessionLabel, _b.SessionLabel);
        var tables = SuspensionChartBuilder.StatsTables(_resultA, _a.SessionLabel);
        if (_resultB is not null)
        {
            tables.AddRange(SuspensionChartBuilder.StatsTables(_resultB, _b.SessionLabel));
        }

        Stats = tables;
    }
}
