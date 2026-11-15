using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>
/// Remembers the last few <see cref="DriverConsistency.Analyze"/> results, so
/// that everything that needs a session's corners (the Driver tab, the region
/// list) runs the kernel once for the same session, laps, channels and
/// thresholds, and so talks about the same corners. Safe to call from several
/// threads: a second caller with the same inputs waits for the first one's
/// result instead of computing it again.
/// </summary>
public sealed class DriverConsistencyCache
{
    /// <summary>Results kept: the Driver tab's laps and the region list's can differ, each for both sessions' worth of switching back and forth.</summary>
    private const int Capacity = 4;

    private readonly object _gate = new();
    private readonly List<(LogFile Log, string Inputs, Lazy<DriverConsistencyResult> Result)> _entries = new();
    private int _computations;

    /// <summary>How many times the kernel has actually run.</summary>
    public int Computations => Volatile.Read(ref _computations);

    /// <summary><see cref="DriverConsistency.Analyze"/>, computed once per distinct set of inputs.</summary>
    public DriverConsistencyResult Analyze(
        LogFile log,
        IReadOnlyList<int> selectedLaps,
        IReadOnlyDictionary<string, string> channelNames,
        double cornerThreshold,
        double throttleThreshold,
        double sustainTimeMs,
        CancellationToken cancellation = default)
    {
        var inputs = string.Join(
            '\n',
            string.Join(',', selectedLaps.Distinct().Order()), // the kernel reads them as a set
            string.Join(';', channelNames.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}")),
            cornerThreshold.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            throttleThreshold.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            sustainTimeMs.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        while (true)
        {
            cancellation.ThrowIfCancellationRequested();
            Lazy<DriverConsistencyResult> entry;
            lock (_gate)
            {
                var found = _entries.FindIndex(e => ReferenceEquals(e.Log, log) && e.Inputs == inputs);
                if (found >= 0)
                {
                    entry = _entries[found].Result;
                    _entries.RemoveAt(found);
                }
                else
                {
                    entry = new Lazy<DriverConsistencyResult>(() =>
                    {
                        Interlocked.Increment(ref _computations);
                        return DriverConsistency.Analyze(log, selectedLaps, channelNames, cornerThreshold, throttleThreshold, sustainTimeMs, cancellation);
                    });
                }

                _entries.Insert(0, (log, inputs, entry)); // most recently used first
                if (_entries.Count > Capacity)
                {
                    _entries.RemoveAt(Capacity);
                }
            }

            try
            {
                return entry.Value;
            }
            catch (OperationCanceledException) when (!cancellation.IsCancellationRequested)
            {
                // The caller that was computing this gave up; this one has not: compute it again.
                Drop(entry);
            }
            catch
            {
                Drop(entry); // a failure is not a result to remember
                throw;
            }
        }
    }

    private void Drop(Lazy<DriverConsistencyResult> entry)
    {
        lock (_gate)
        {
            _entries.RemoveAll(e => ReferenceEquals(e.Result, entry));
        }
    }
}
