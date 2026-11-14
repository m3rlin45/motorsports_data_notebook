using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// The analysis loop every tab shares: what is displayed when requests
/// overlap, when there is nothing to analyse, and when a run fails.
/// </summary>
public class AnalysisTabTests
{
    /// <summary>A tab whose analysis blocks until released, so a test can change the inputs mid-run.</summary>
    private sealed class GatedTab : AnalysisTabViewModel<GatedTab.Input, string>
    {
        public sealed record Input(string Value);

        public string? Next { get; set; } = "first";
        public string? HintText { get; set; }
        public bool Fail { get; set; }
        public SemaphoreSlim Started { get; } = new(0);
        public SemaphoreSlim Release { get; } = new(0);
        public List<string> Displayed { get; } = new();
        public int Cleared { get; private set; }
        public int Cancelled { get; private set; }

        public override string Header => "Gated";

        protected override Input? GetParameters() => Next is null ? (HintText is null ? null : Hint(HintText)) : new Input(Next);

        protected override string Analyze(Input parameters, CancellationToken cancellation)
        {
            Started.Release();
            Release.Wait(TestContext.Current.CancellationToken);
            if (cancellation.IsCancellationRequested)
            {
                Cancelled++;
            }

            if (Fail)
            {
                throw new InvalidOperationException("no corners");
            }

            return parameters.Value.ToUpperInvariant();
        }

        protected override void Display(Input parameters, string result) => Displayed.Add(result);

        protected override void ClearDisplay() => Cleared++;
    }

    [Fact]
    public async Task A_result_overtaken_by_a_newer_request_is_never_displayed()
    {
        var tab = new GatedTab();
        var completions = 0;
        tab.AnalysisCompleted += () => completions++;

        var run = tab.RunNowAsync();
        await tab.Started.WaitAsync(TestContext.Current.CancellationToken);
        Assert.True(tab.IsAnalyzing);
        Assert.Equal(StatusKind.Busy, tab.StatusKind);

        // The inputs change while the first analysis is still computing.
        tab.Next = "second";
        var same = tab.RunNowAsync();
        Assert.Same(run, same); // callers await the chain, which ends with the rerun

        tab.Release.Release(2);
        await run;

        Assert.Equal(new[] { "SECOND" }, tab.Displayed);
        Assert.Equal(1, tab.Cancelled); // the stale run was told to stop
        Assert.Equal(1, completions);
        Assert.Equal("Analysis complete", tab.Status);
        Assert.Equal(StatusKind.Done, tab.StatusKind);
        Assert.Null(tab.EmptyMessage);
        Assert.False(tab.IsAnalyzing);
    }

    [Fact]
    public async Task Nothing_to_analyse_clears_the_display_and_says_why()
    {
        var tab = new GatedTab();
        tab.Release.Release();
        await tab.RunNowAsync();
        Assert.Single(tab.Displayed);

        tab.Next = null;
        tab.HintText = "Select laps to analyze";
        await tab.RunNowAsync();

        // The old chart must not stay up as if it were the answer.
        Assert.Equal(1, tab.Cleared);
        Assert.Equal("Select laps to analyze", tab.Status);
        Assert.Equal(StatusKind.Hint, tab.StatusKind);
        Assert.Equal("Select laps to analyze", tab.EmptyMessage);

        // A tab that returns null without a reason still gets one.
        tab.HintText = null;
        await tab.RunNowAsync();
        Assert.Equal(AnalysisTabViewModel.NoSessionHint, tab.Status);
    }

    [Fact]
    public async Task A_failed_analysis_is_an_error_status_and_clears_the_display()
    {
        var tab = new GatedTab { Fail = true };
        tab.Release.Release();

        await tab.RunNowAsync();

        Assert.Empty(tab.Displayed);
        Assert.Equal(1, tab.Cleared);
        Assert.Equal("Error: no corners", tab.Status);
        Assert.True(tab.IsError);
        Assert.Equal("Error: no corners", tab.EmptyMessage);
        Assert.False(tab.IsAnalyzing);
    }
}
