using InfernoAnalyzer.App;

namespace InfernoAnalyzer.Tests;

/// <summary>The hang report's stack capture (Windows x64; elsewhere there is nothing to capture).</summary>
public class HangWatchTests
{
    [Fact]
    public void The_stack_of_a_blocked_thread_is_captured_as_modules_and_offsets()
    {
        Assert.SkipUnless(HangWatch.IsSupported, "the stack walk uses the Windows x64 unwinder");

        using var ready = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        uint id = 0;
        var blocked = new Thread(() =>
        {
            id = HangWatch.CurrentThreadId();
            ready.Set();
            release.Wait();
        });
        blocked.Start();
        ready.Wait(TestContext.Current.CancellationToken);
        Thread.Sleep(200); // let it get from Set into the wait

        var report = HangWatch.CaptureStack(id);
        release.Set();
        blocked.Join();

        Console.Error.WriteLine(report); // in the CI log, for a look at what a report contains
        var frames = report.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.True(frames.Length >= 4, report);
        Assert.All(frames, frame => Assert.Matches(@"^#\d\d \S+", frame));
        Assert.Contains("ntdll.dll+0x", report, StringComparison.OrdinalIgnoreCase); // a waiting thread is inside the system's wait
        Assert.Contains("kernel32.dll+0x", report, StringComparison.OrdinalIgnoreCase); // and started in the system's thread start
    }

    [Fact]
    public void Elsewhere_a_capture_says_so_instead_of_failing()
    {
        Assert.SkipWhen(HangWatch.IsSupported, "covered by the capture test");
        Assert.Equal("(not supported on this platform)", HangWatch.CaptureStack(0));
    }
}
