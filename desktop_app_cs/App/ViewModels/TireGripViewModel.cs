using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>What the tire grip is plotted against.</summary>
public enum TireMetric
{
    /// <summary>TPMS pressure.</summary>
    Pressure,

    /// <summary>TPMS temperature.</summary>
    Temperature,
}

/// <summary>Tire Grip tab config: acceleration and TPMS channels per session.</summary>
public sealed class TireGripConfigViewModel : ChannelConfigViewModel
{
    /// <summary>Create the config.</summary>
    public TireGripConfigViewModel()
        : base(new[]
        {
            ("lateral_g", "Lat G"), ("inline_g", "Inline G"),
            ("tpms_press_fl", "Press FL"), ("tpms_press_fr", "Press FR"), ("tpms_press_rl", "Press RL"), ("tpms_press_rr", "Press RR"),
            ("tpms_temp_fl", "Temp FL"), ("tpms_temp_fr", "Temp FR"), ("tpms_temp_rl", "Temp RL"), ("tpms_temp_rr", "Temp RR"),
        })
    {
    }
}

/// <summary>Builds the tire-grip figure and tables.</summary>
public static class TireGripChartBuilder
{
    private static readonly string[] Titles = { "Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)" };

    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    /// <summary>The 2×2 bucketed-percentile lines, one or two sessions per panel.</summary>
    public static LineGridSpec Build(TireGripResult a, TireGripResult? b, string labelA, string labelB)
    {
        var metricLabel = a.MetricMode == "pressure" ? "Pressure" : "Temperature";
        var xLabel = $"{metricLabel} ({a.MetricUnit})";
        var cornersA = a.Corners.ToList();
        var cornersB = b?.Corners.ToList();
        var panels = new List<LinePanelSpec>();
        for (var i = 0; i < 4; i++)
        {
            var ca = cornersA[i];
            var series = new List<LineSeriesSpec>();
            if (cornersB is null)
            {
                series.Add(new LineSeriesSpec(string.Empty, ca.BucketCenters, ca.BucketValues, ca.BucketCounts, ChartColors.SessionA));
                panels.Add(new LinePanelSpec(Titles[i], series, $"n = {ca.BucketCounts.Sum()}"));
            }
            else
            {
                var cb = cornersB[i];
                series.Add(new LineSeriesSpec(labelA, ca.BucketCenters, ca.BucketValues, ca.BucketCounts, ChartColors.SessionA));
                series.Add(new LineSeriesSpec(labelB, cb.BucketCenters, cb.BucketValues, cb.BucketCounts, ChartColors.SessionB));
                panels.Add(new LinePanelSpec(Titles[i], series));
            }
        }

        var title = b is null ? $"Tire Grip: {labelA}" : $"Comparison: {labelA} vs {labelB}";
        var hover = "{0}\n" + metricLabel + ": {1:0.0} " + a.MetricUnit + "\nTotal G: {2:0.00}\nn = {3}";
        return new LineGridSpec(title, panels, 2, xLabel, "Total G", hover);
    }

    /// <summary><c>format_tire_grip_stats_table</c> plus the comparison and detail tables of the stats panel.</summary>
    public static List<StatsTable> Tables(TireGripResult a, TireGripResult? b, string labelA, string labelB)
    {
        var metricLabel = a.MetricMode == "pressure" ? "Pressure" : "Temperature";
        var unit = a.MetricUnit.Length > 0 ? $" ({a.MetricUnit})" : string.Empty;
        var accel = a.AccelUnit.Length > 0 ? $" ({a.AccelUnit})" : string.Empty;
        var ca = a.Corners.ToList();
        var tables = new List<StatsTable>();
        if (b is null)
        {
            tables.Add(new StatsTable(
                $"Tire grip — {labelA}",
                new[] { "Corner", $"Mean Accel{accel}", $"Std Accel{accel}", $"Mean {metricLabel}{unit}", $"Std {metricLabel}{unit}" },
                ca.Select(c => (IReadOnlyList<string>)new[] { c.CornerName, F(c.MeanG, "0.000"), F(c.StdG, "0.000"), F(c.MeanMetric, "0.0"), F(c.StdMetric, "0.0") }).ToList()));
        }
        else
        {
            var cb = b.Corners.ToList();
            tables.Add(new StatsTable(
                "Mean total G",
                new[] { "Corner", $"A{accel}", $"B{accel}", "Δ" },
                ca.Select((c, i) => (IReadOnlyList<string>)new[] { c.CornerName, F(c.MeanG, "0.000"), F(cb[i].MeanG, "0.000"), F(cb[i].MeanG - c.MeanG, "+0.000;-0.000") }).ToList()));
            tables.Add(new StatsTable(
                $"{metricLabel} details",
                new[] { "Corner", $"A Mean{unit}", $"B Mean{unit}", "Δ Mean", $"A Std{unit}", $"B Std{unit}", "Δ Std" },
                ca.Select((c, i) => (IReadOnlyList<string>)new[] { c.CornerName, F(c.MeanMetric, "0.0"), F(cb[i].MeanMetric, "0.0"), F(cb[i].MeanMetric - c.MeanMetric, "+0.0;-0.0"), F(c.StdMetric, "0.0"), F(cb[i].StdMetric, "0.0"), F(cb[i].StdMetric - c.StdMetric, "+0.0;-0.0") }).ToList()));
        }

        return tables;
    }
}

/// <summary>The Tire Grip tab: total G against tire pressure or temperature per corner.</summary>
public sealed partial class TireGripViewModel : AnalysisTabViewModel<TireGripViewModel.Parameters, TireGripViewModel.Result>
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;

    /// <summary>Create the tab over the two session slots.</summary>
    public TireGripViewModel(SessionViewModel a, SessionViewModel b)
    {
        _a = a;
        _b = b;
        TireConfig = new TireGripConfigViewModel();
        TireConfig.Changed += OnSelectionChanged;
        Percentile = new NumericFieldViewModel("Percentile", "0.###", min: 0, max: 100, minExclusive: true, onChanged: OnSelectionChanged);
        Percentile.Set(99.9);
    }

    /// <inheritdoc />
    public override string Header => "Tire Grip";

    /// <summary>Channel mapping.</summary>
    public TireGripConfigViewModel TireConfig { get; }

    /// <inheritdoc />
    public override object? Config => TireConfig;

    /// <inheritdoc />
    public override bool HasStats => true;

    /// <summary>What the grip is plotted against.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPressure))]
    [NotifyPropertyChangedFor(nameof(IsTemperature))]
    private TireMetric _metric = TireMetric.Pressure;

    /// <summary>Radio binding.</summary>
    public bool IsPressure
    {
        get => Metric == TireMetric.Pressure;
        set
        {
            if (value)
            {
                Metric = TireMetric.Pressure;
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsTemperature
    {
        get => Metric == TireMetric.Temperature;
        set
        {
            if (value)
            {
                Metric = TireMetric.Temperature;
            }
        }
    }

    partial void OnMetricChanged(TireMetric value) => OnSelectionChanged();

    /// <summary>Percentile of total G per bucket.</summary>
    public NumericFieldViewModel Percentile { get; }

    /// <summary>The figure.</summary>
    [ObservableProperty]
    private LineGridSpec? _spec;

    /// <summary>Stats tables.</summary>
    [ObservableProperty]
    private IReadOnlyList<StatsTable> _stats = Array.Empty<StatsTable>();

    /// <inheritdoc />
    public override void OnSessionLoaded()
    {
        TireConfig.OnSessionsChanged(_a, _b);
        base.OnSessionLoaded();
    }

    /// <summary>Everything one analysis reads, snapshotted on the UI thread.</summary>
    public sealed record Parameters(
        LogFile A,
        List<int> LapsA,
        Dictionary<string, string> NamesA,
        string LabelA,
        LogFile? B,
        List<int> LapsB,
        Dictionary<string, string> NamesB,
        string LabelB,
        TireMetric Metric,
        double Percentile);

    /// <summary>The grip curves of Session A and, in a comparison, Session B.</summary>
    public sealed record Result(TireGripResult A, TireGripResult? B);

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

        if (Percentile.Error is { } error)
        {
            return Hint(error);
        }

        // A comparison session with no laps ticked is simply not compared.
        var lapsB = _b.Log is null ? new List<int>() : _b.SelectedIn(Scope);
        var logB = lapsB.Count > 0 ? _b.Log : null;
        return new Parameters(
            _a.Log,
            lapsA,
            TireConfig.GetChannelNames(SessionSlot.A),
            _a.SessionLabel,
            logB,
            lapsB,
            TireConfig.GetChannelNames(SessionSlot.B),
            _b.SessionLabel,
            Metric,
            Percentile.Value ?? 99.9);
    }

    /// <inheritdoc />
    protected override Result Analyze(Parameters p, CancellationToken cancellation)
    {
        var mode = p.Metric == TireMetric.Pressure ? "pressure" : "temperature"; // the kernel keeps Python's mode names
        var a = TireGrip.AnalyzeTireGripMultiLap(p.A, p.LapsA, p.NamesA, mode, percentile: p.Percentile, cancellation: cancellation);
        var b = p.B is null ? null : TireGrip.AnalyzeTireGripMultiLap(p.B, p.LapsB, p.NamesB, mode, percentile: p.Percentile, cancellation: cancellation);
        return new Result(a, b);
    }

    /// <inheritdoc />
    protected override void Display(Parameters p, Result result)
    {
        Spec = TireGripChartBuilder.Build(result.A, result.B, p.LabelA, p.LabelB);
        Stats = TireGripChartBuilder.Tables(result.A, result.B, p.LabelA, p.LabelB);
        if (result.A.Corners.All(c => c.TotalG.Length == 0))
        {
            SetStatus("No data: check the acceleration and TPMS channel names under Channels", StatusKind.Hint);
        }
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
        state["metric"] = Metric.ToString();
        if (Percentile.IsValid)
        {
            state["percentile"] = Percentile.Text;
        }
    }

    /// <inheritdoc />
    public override void RestoreState(IReadOnlyDictionary<string, string> state)
    {
        base.RestoreState(state);
        if (state.TryGetValue("metric", out var metric) && Enum.TryParse<TireMetric>(metric, out var m))
        {
            Metric = m;
        }

        if (state.TryGetValue("percentile", out var text) && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) && p is > 0 and <= 100)
        {
            Percentile.Set(p);
        }
    }
}
