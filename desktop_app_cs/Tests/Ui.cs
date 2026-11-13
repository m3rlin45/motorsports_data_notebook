using Avalonia.Headless;

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
    public static Task Run(Action test) => Session.Value.Dispatch(test, TestContext.Current.CancellationToken);

    /// <summary>Run an asynchronous test body on the UI thread.</summary>
    public static Task Run(Func<Task> test) => Session.Value.Dispatch(
        async () =>
        {
            await test();
            return true;
        },
        TestContext.Current.CancellationToken);
}
