using InfernoAnalyzer.Core.Analysis;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The shared driver-consistency results: one run of the kernel per distinct set of inputs.</summary>
public class DriverConsistencyCacheTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static IReadOnlyDictionary<string, string> Names()
    {
        var names = new Dictionary<string, string>(ChannelKeys.DefaultChannelNames);
        foreach (var (key, value) in BuiltinOnly.GetProfileForLog(Sample.Value.Metadata, Path.GetFileName(SessionParityTests.SamplePath()))!.ChannelNames)
        {
            names[key] = value;
        }

        return names;
    }

    [Fact]
    public void The_same_inputs_are_computed_once_and_different_ones_separately()
    {
        var log = Sample.Value;
        var laps = LapSelection.GetTopLaps(log.Laps).Select(l => l.Num).Take(3).ToList();
        var cache = new DriverConsistencyCache();
        var ct = TestContext.Current.CancellationToken;

        var first = cache.Analyze(log, laps, Names(), Corners.DefaultThreshold, 90, 500, ct);
        var again = cache.Analyze(log, laps.ToList(), Names(), Corners.DefaultThreshold, 90, 500, ct);
        Assert.Same(first, again);
        Assert.Equal(1, cache.Computations);

        var fewer = cache.Analyze(log, laps.Take(2).ToList(), Names(), Corners.DefaultThreshold, 90, 500, ct);
        Assert.NotSame(first, fewer);
        Assert.Equal(2, cache.Computations);
        Assert.Same(first, cache.Analyze(log, laps, Names(), Corners.DefaultThreshold, 90, 500, ct)); // both are kept
        Assert.Equal(2, cache.Computations);

        cache.Analyze(log, laps, Names(), Corners.DefaultThreshold, 80, 500, ct); // another threshold is another analysis
        Assert.Equal(3, cache.Computations);
    }

    [Fact]
    public void A_failed_or_cancelled_run_is_not_remembered()
    {
        var log = Sample.Value;
        var laps = LapSelection.GetTopLaps(log.Laps).Select(l => l.Num).Take(2).ToList();
        var cache = new DriverConsistencyCache();

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => cache.Analyze(log, laps, Names(), Corners.DefaultThreshold, 90, 500, cancelled.Token));

        // A threshold no corner reaches: the kernel's "no corners" answer, every time it is asked.
        Assert.Throws<InvalidOperationException>(() => cache.Analyze(log, laps, Names(), 10, 90, 500, TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => cache.Analyze(log, laps, Names(), 10, 90, 500, TestContext.Current.CancellationToken));
        Assert.Equal(2, cache.Computations);

        Assert.NotNull(cache.Analyze(log, laps, Names(), Corners.DefaultThreshold, 90, 500, TestContext.Current.CancellationToken));
        Assert.Equal(3, cache.Computations);
    }
}
