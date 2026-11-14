using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>What the status line is reporting, so the shell can colour and badge it.</summary>
public enum StatusKind
{
    /// <summary>Nothing to say yet.</summary>
    Idle,

    /// <summary>The tab needs something from the user before it can analyse (a session, laps, a channel name).</summary>
    Hint,

    /// <summary>An analysis is running.</summary>
    Busy,

    /// <summary>The last analysis finished.</summary>
    Done,

    /// <summary>The last analysis failed.</summary>
    Error,
}

/// <summary>Which of a session's lap selections a tab works on.</summary>
public enum LapScope
{
    /// <summary>The laps aggregated by the statistical tabs (default: within 103 % of the best).</summary>
    Analysis,

    /// <summary>The laps overlaid as traces (default: the best two).</summary>
    Traces,
}

/// <summary>A command a tab contributes to the View menu and the window's key bindings while it is active.</summary>
/// <param name="Header">Menu text, with an access key (e.g. "_Find channel").</param>
/// <param name="Gesture">Key gesture in Avalonia's syntax with "Cmd" for the platform's command modifier, e.g. "Cmd+F", "F9"; null for none.</param>
/// <param name="Command">What it does.</param>
public sealed record TabCommand(string Header, string? Gesture, ICommand Command);

/// <summary>
/// Shared tab behaviour from the Python <c>BaseAnalysisTab</c>: a 300 ms
/// debounce after any change, one analysis at a time on a worker thread,
/// a stale flag so a hidden tab recomputes when shown, and a status line.
/// The compute/display contract is in
/// <see cref="AnalysisTabViewModel{TParameters, TResult}"/>; this base is
/// what the shell binds to.
/// </summary>
public abstract partial class AnalysisTabViewModel : ObservableObject
{
    /// <summary>The hint every tab shows before a session is loaded.</summary>
    public const string NoSessionHint = "Open a session or drop a telemetry file here";

    private CancellationTokenSource? _debounce;
    private CancellationTokenSource? _analysis;
    private Task? _running;
    private bool _rerun;
    private bool _stale;

    /// <summary>Debounce interval; tests shorten it.</summary>
    public TimeSpan Debounce { get; set; } = TimeSpan.FromMilliseconds(300);

    /// <summary>Tab header.</summary>
    public abstract string Header { get; }

    /// <summary>The tab's channel configuration, edited in the vehicle profile window (null: none).</summary>
    public virtual object? Config => null;

    /// <summary>Which lap selection of the sessions the tab analyses.</summary>
    public virtual LapScope Scope => LapScope.Analysis;

    /// <summary>Whether the tab draws one coloured trace per lap, so the lap list should show the colour keys.</summary>
    public virtual bool UsesLapColors => false;

    /// <summary>Whether the tab has a statistics pane (<see cref="ShowStats"/>).</summary>
    public virtual bool HasStats => false;

    /// <summary>Commands for the View menu and key bindings while the tab is active.</summary>
    public virtual IReadOnlyList<TabCommand> Commands => Array.Empty<TabCommand>();

    /// <summary>Status line.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsError))]
    private string _status = NoSessionHint;

    /// <summary>What <see cref="Status"/> is reporting.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsError))]
    private StatusKind _statusKind = StatusKind.Hint;

    /// <summary>Whether the status is an error.</summary>
    public bool IsError => StatusKind == StatusKind.Error;

    /// <summary>
    /// Why there is nothing to show (no session, no laps, a failed run), for
    /// the chart area's empty state; null while a result is displayed.
    /// </summary>
    [ObservableProperty]
    private string? _emptyMessage = NoSessionHint;

    /// <summary>True while an analysis runs.</summary>
    [ObservableProperty]
    private bool _isAnalyzing;

    /// <summary>Whether the statistics pane is shown.</summary>
    [ObservableProperty]
    private bool _showStats;

    /// <summary>Whether this tab is the visible one.</summary>
    public bool IsActive { get; private set; }

    /// <summary>Raised on the UI thread after a successful analysis.</summary>
    public event Action? AnalysisCompleted;

    /// <summary>Set the status line and its kind together.</summary>
    public void SetStatus(string text, StatusKind kind)
    {
        Status = text;
        StatusKind = kind;
    }

    /// <summary>Called when a session (re)loads or closes: forget results, recompute when visible.</summary>
    public virtual void OnSessionLoaded()
    {
        _stale = true;
        if (IsActive)
        {
            RequestAnalysis();
        }
    }

    /// <summary>Called when the tab becomes visible or hidden.</summary>
    public void SetActive(bool active)
    {
        IsActive = active;
        if (active && _stale)
        {
            _stale = false;
            RequestAnalysis();
        }
    }

    /// <summary>Lap selection or config changed: analyse after the debounce.</summary>
    public void OnSelectionChanged()
    {
        if (IsActive)
        {
            RequestAnalysis();
        }
        else
        {
            _stale = true;
        }
    }

    /// <summary>Schedule an analysis after <see cref="Debounce"/>, coalescing bursts.</summary>
    public void RequestAnalysis()
    {
        _stale = false;
        _debounce?.Cancel();
        var cts = new CancellationTokenSource();
        _debounce = cts;
        _ = DebouncedRunAsync(cts.Token);
    }

    private async Task DebouncedRunAsync(CancellationToken token)
    {
        try
        {
            await Task.Delay(Debounce, token);
        }
        catch (TaskCanceledException)
        {
            return;
        }

        await RunNowAsync();
    }

    /// <summary>
    /// Run the analysis now. If one is in flight it is cancelled and a rerun
    /// flagged; the in-flight chain is returned, which only completes after
    /// the rerun has finished, so awaiting callers always see the final result.
    /// </summary>
    public Task RunNowAsync()
    {
        if (_running is { } chain)
        {
            _rerun = true; // a change arrived mid-analysis; abandon it and run again
            _analysis?.Cancel();
            return chain;
        }

        var run = RunChainAsync();
        if (!run.IsCompleted)
        {
            _running = run;
        }

        return run;
    }

    private async Task RunChainAsync()
    {
        try
        {
            do
            {
                _rerun = false;
                using var cts = new CancellationTokenSource();
                _analysis = cts;
                try
                {
                    await RunOnceAsync(cts.Token, () => _rerun);
                }
                finally
                {
                    _analysis = null;
                }
            }
            while (_rerun);
        }
        finally
        {
            _running = null;
        }
    }

    /// <summary>
    /// One pass: snapshot the inputs, compute off the UI thread, display.
    /// <paramref name="superseded"/> turns true when a newer request arrived
    /// while computing; the pass must then display nothing.
    /// </summary>
    private protected abstract Task RunOnceAsync(CancellationToken cancellation, Func<bool> superseded);

    /// <summary>Raise <see cref="AnalysisCompleted"/>.</summary>
    private protected void RaiseAnalysisCompleted() => AnalysisCompleted?.Invoke();

    /// <summary>Save view state worth keeping between runs of the app (keys are the tab's own).</summary>
    public virtual void SaveState(IDictionary<string, string> state)
    {
        if (HasStats)
        {
            state["stats"] = ShowStats ? "1" : "0";
        }
    }

    /// <summary>Restore what <see cref="SaveState"/> wrote; missing or malformed entries are ignored.</summary>
    public virtual void RestoreState(IReadOnlyDictionary<string, string> state)
    {
        if (HasStats && state.TryGetValue("stats", out var stats))
        {
            ShowStats = stats == "1";
        }
    }
}

/// <summary>
/// The three-step contract of an analysis tab: <see cref="GetParameters"/>
/// snapshots the inputs on the UI thread, <see cref="Analyze"/> computes a
/// result from that snapshot alone on a worker thread, and
/// <see cref="Display"/> shows it back on the UI thread. Nothing crosses
/// threads except the two immutable values, and a result that was overtaken
/// by a newer request is dropped rather than drawn.
/// </summary>
/// <typeparam name="TParameters">Immutable snapshot of everything the analysis reads.</typeparam>
/// <typeparam name="TResult">What the analysis produces.</typeparam>
public abstract class AnalysisTabViewModel<TParameters, TResult> : AnalysisTabViewModel
    where TParameters : class
{
    /// <summary>
    /// Read the inputs from the sessions and config. Return
    /// <see cref="Hint"/> when there is nothing to analyse.
    /// </summary>
    protected abstract TParameters? GetParameters();

    /// <summary>Compute on a worker thread. Check <paramref name="cancellation"/> between laps.</summary>
    protected abstract TResult Analyze(TParameters parameters, CancellationToken cancellation);

    /// <summary>Show <paramref name="result"/> (UI thread). Labels and names come from <paramref name="parameters"/>, not from the live sessions.</summary>
    protected abstract void Display(TParameters parameters, TResult result);

    /// <summary>Drop what is displayed (UI thread): the inputs no longer produce a result.</summary>
    protected abstract void ClearDisplay();

    private bool _hinted;

    /// <summary>Report why there is nothing to analyse; returns null for <see cref="GetParameters"/> to return.</summary>
    protected TParameters? Hint(string message)
    {
        _hinted = true;
        SetStatus(message, StatusKind.Hint);
        return null;
    }

    private protected sealed override async Task RunOnceAsync(CancellationToken cancellation, Func<bool> superseded)
    {
        _hinted = false;
        var parameters = GetParameters();
        if (parameters is null)
        {
            if (!_hinted)
            {
                SetStatus(NoSessionHint, StatusKind.Hint);
            }

            ClearDisplay();
            EmptyMessage = Status;
            return;
        }

        IsAnalyzing = true;
        SetStatus("Analyzing...", StatusKind.Busy);
        try
        {
            var result = await Task.Run(() => Analyze(parameters, cancellation), cancellation);
            if (superseded())
            {
                return; // the inputs changed while computing; the rerun shows the current ones
            }

            EmptyMessage = null;
            Display(parameters, result);
            if (Status == "Analyzing...")
            {
                SetStatus("Analysis complete", StatusKind.Done); // unless Display() reported something more specific
            }
            else if (StatusKind == StatusKind.Busy)
            {
                StatusKind = StatusKind.Done;
            }

            RaiseAnalysisCompleted();
        }
        catch (OperationCanceledException) when (superseded())
        {
            // Abandoned for a newer request.
        }
        catch (Exception e)
        {
            SetStatus($"Error: {e.Message}", StatusKind.Error);
            ClearDisplay();
            EmptyMessage = Status;
        }
        finally
        {
            IsAnalyzing = false;
        }
    }
}
