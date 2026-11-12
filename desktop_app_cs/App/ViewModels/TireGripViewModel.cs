using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.App.ViewModels;

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
    private const string Blue = "#4682b4";
    private const string Orange = "#ff8c00";
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
                series.Add(new LineSeriesSpec(string.Empty, ca.BucketCenters, ca.BucketValues, ca.BucketCounts, Blue));
                panels.Add(new LinePanelSpec(Titles[i], series, $"n = {ca.BucketCounts.Sum()}"));
            }
            else
            {
                var cb = cornersB[i];
                series.Add(new LineSeriesSpec(labelA, ca.BucketCenters, ca.BucketValues, ca.BucketCounts, Blue));
                series.Add(new LineSeriesSpec(labelB, cb.BucketCenters, cb.BucketValues, cb.BucketCounts, Orange));
                panels.Add(new LinePanelSpec(Titles[i], series));
            }
        }

        var title = b is null ? $"Tire Grip: {labelA}" : $"Comparison: {labelA} vs {labelB}";
        return new LineGridSpec(title, panels, 2, xLabel, "Total G");
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
public sealed partial class TireGripViewModel : AnalysisTabViewModel
{
    private readonly SessionViewModel _a;
    private readonly SessionViewModel _b;
    private TireGripResult? _resultA;
    private TireGripResult? _resultB;

    /// <summary>Create the tab over the two session slots.</summary>
    public TireGripViewModel(SessionViewModel a, SessionViewModel b)
    {
        _a = a;
        _b = b;
        TireConfig = new TireGripConfigViewModel();
        TireConfig.Changed += OnSelectionChanged;
    }

    /// <inheritdoc />
    public override string Header => "Tire Grip";

    /// <summary>Channel mapping.</summary>
    public TireGripConfigViewModel TireConfig { get; }

    /// <inheritdoc />
    public override object? Config => TireConfig;

    /// <summary>"pressure" or "temperature".</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPressure))]
    [NotifyPropertyChangedFor(nameof(IsTemperature))]
    private string _metricMode = "pressure";

    /// <summary>Radio binding.</summary>
    public bool IsPressure
    {
        get => MetricMode == "pressure";
        set
        {
            if (value)
            {
                MetricMode = "pressure";
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsTemperature
    {
        get => MetricMode == "temperature";
        set
        {
            if (value)
            {
                MetricMode = "temperature";
            }
        }
    }

    partial void OnMetricModeChanged(string value) => OnSelectionChanged();

    /// <summary>Percentile of total G per bucket, as text.</summary>
    [ObservableProperty]
    private string _percentile = "99.9";

    partial void OnPercentileChanged(string value) => OnSelectionChanged();

    /// <summary>The figure.</summary>
    [ObservableProperty]
    private LineGridSpec? _spec;

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
            TireConfig.UpdateAvailableChannels("A", _a.Log.ChannelNames.ToList());
            if (_a.Profile is { } pa)
            {
                TireConfig.SetFromProfile(pa, "A");
            }
        }

        if (_b.Log is not null)
        {
            TireConfig.HasSessionB = true;
            TireConfig.UpdateAvailableChannels("B", _b.Log.ChannelNames.ToList());
            if (_b.Profile is { } pb && (_b.LoggerId, _b.VehicleName) != (_a.LoggerId, _a.VehicleName))
            {
                TireConfig.SyncWithA = false;
                TireConfig.SetFromProfile(pb, "B");
            }
        }

        _resultA = null;
        _resultB = null;
        base.OnSessionLoaded();
    }

    private sealed record Parameters(LogFile A, List<int> LapsA, Dictionary<string, string> NamesA, LogFile? B, List<int> LapsB, Dictionary<string, string> NamesB, string Mode, double Percentile);

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

        var percentile = double.TryParse(Percentile, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) && p is > 0 and <= 100 ? p : 99.9;
        return new Parameters(_a.Log, lapsA, TireConfig.GetChannelNames("A"), _b.Log, lapsB, TireConfig.GetChannelNames("B"), MetricMode, percentile);
    }

    /// <inheritdoc />
    protected override void Analyze(object parameters)
    {
        var p = (Parameters)parameters;
        _resultA = TireGrip.AnalyzeTireGripMultiLap(p.A, p.LapsA, p.NamesA, p.Mode, percentile: p.Percentile);
        _resultB = p.B is not null && p.LapsB.Count > 0 ? TireGrip.AnalyzeTireGripMultiLap(p.B, p.LapsB, p.NamesB, p.Mode, percentile: p.Percentile) : null;
    }

    /// <inheritdoc />
    protected override void Display()
    {
        if (_resultA is null)
        {
            return;
        }

        Spec = TireGripChartBuilder.Build(_resultA, _resultB, _a.SessionLabel, _b.SessionLabel);
        Stats = TireGripChartBuilder.Tables(_resultA, _resultB, _a.SessionLabel, _b.SessionLabel);
        if (_resultA.Corners.All(c => c.TotalG.Length == 0))
        {
            Status = "No data: check the acceleration and TPMS channel names";
        }
    }
}
