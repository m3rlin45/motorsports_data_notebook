using CommunityToolkit.Mvvm.ComponentModel;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>
/// Shared tab behaviour from the Python <c>BaseAnalysisTab</c>: a 300 ms
/// debounce after any change, one analysis at a time on a worker thread,
/// a stale flag so a hidden tab recomputes when shown, and a status line.
/// </summary>
public abstract partial class AnalysisTabViewModel : ObservableObject
{
    private CancellationTokenSource? _debounce;
    private Task? _running;
    private bool _rerun;
    private bool _stale;

    /// <summary>Debounce interval; tests shorten it.</summary>
    public TimeSpan Debounce { get; set; } = TimeSpan.FromMilliseconds(300);

    /// <summary>Tab header.</summary>
    public abstract string Header { get; }

    /// <summary>The tab's configuration view model, shown in the top panel while the tab is active (null: none).</summary>
    public virtual object? Config => null;

    /// <summary>Status line shown in the config area.</summary>
    [ObservableProperty]
    private string _status = string.Empty;

    /// <summary>True while an analysis runs.</summary>
    [ObservableProperty]
    private bool _isAnalyzing;

    /// <summary>Whether this tab is the visible one.</summary>
    public bool IsActive { get; private set; }

    /// <summary>Raised on the UI thread after a successful analysis.</summary>
    public event Action? AnalysisCompleted;

    /// <summary>Called when a session (re)loads: forget results, recompute when visible.</summary>
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
    /// Run the analysis now. If one is in flight, flag a rerun and return the
    /// in-flight chain, which only completes after the rerun has finished, so
    /// awaiting callers always see the final result.
    /// </summary>
    public Task RunNowAsync()
    {
        if (_running is { } chain)
        {
            _rerun = true; // a change arrived mid-analysis; run again afterwards
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
                var parameters = GetParameters();
                if (parameters is null)
                {
                    break;
                }

                await RunAsync(parameters);
            }
            while (_rerun);
        }
        finally
        {
            _running = null;
        }
    }

    private async Task RunAsync(object parameters)
    {
        IsAnalyzing = true;
        Status = "Analyzing...";
        try
        {
            await Task.Run(() => Analyze(parameters));
            Display();
            if (Status == "Analyzing...")
            {
                Status = "Analysis complete"; // unless Display() reported something more specific
            }

            AnalysisCompleted?.Invoke();
        }
        catch (Exception e)
        {
            Status = $"Error: {e.Message}";
        }
        finally
        {
            IsAnalyzing = false;
        }
    }

    /// <summary>Read the inputs from the sessions and config; null when nothing to analyse (set <see cref="Status"/> to say why).</summary>
    protected abstract object? GetParameters();

    /// <summary>Compute on a worker thread and store the result.</summary>
    protected abstract void Analyze(object parameters);

    /// <summary>Push the stored result into the view (UI thread).</summary>
    protected abstract void Display();
}
