using Avalonia.Headless;
using InfernoAnalyzer.App;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Runs a test body on the UI thread of the headless Avalonia app (<see cref="TestAppBuilder"/>),
/// with a fresh application per test. This is what Avalonia.Headless.XUnit's <c>[AvaloniaFact]</c>
/// does, without its reflection-based test discovery, so the tests also run NativeAOT-compiled.
/// </summary>
internal static class Ui
{
    private static readonly Lazy<HeadlessUnitTestSession> Session =
        new(() => HeadlessUnitTestSession.StartNew(typeof(TestAppBuilder), AvaloniaTestIsolationLevel.PerTest));

    /// <summary>Run a synchronous test body on the UI thread.</summary>
    public static Task Run(Action test) => Run(() =>
    {
        test();
        return Task.CompletedTask;
    });

    /// <summary>
    /// Run an asynchronous test body on the UI thread. An error Avalonia logs while it runs fails the
    /// test: Avalonia reports failures inside it (an exception while a frame is drawn, for one) to
    /// its logger and carries on, which would otherwise pass unnoticed.
    /// </summary>
    public static Task Run(Func<Task> test) => Session.Value.Dispatch(
        async () =>
        {
            var errors = new List<string>();
            void Collect(string error)
            {
                lock (errors)
                {
                    errors.Add(error);
                }
            }

            AvaloniaLog.Reported += Collect;
            try
            {
                await test();
                Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            }
            finally
            {
                AvaloniaLog.Reported -= Collect;
            }

            lock (errors)
            {
                Assert.True(errors.Count == 0, "Avalonia logged errors:\n" + string.Join("\n", errors));
            }

            return true;
        },
        TestContext.Current.CancellationToken);
}
