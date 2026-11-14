using Avalonia.Logging;
using InfernoAnalyzer.App;

namespace InfernoAnalyzer.Tests;

/// <summary>What happens to the things Avalonia only logs.</summary>
public class AvaloniaLogTests
{
    private readonly List<(string Context, string Text)> _written = new();
    private readonly List<string> _shown = new();
    private readonly List<string> _reported = new();
    private readonly List<(string Context, string Text)> _platform = new();

    private AvaloniaLog Sink() => new(
        (context, text) => _written.Add((context, text)), _shown.Add, _reported.Add, (context, text) => _platform.Add((context, text)));

    [Fact]
    public void An_exception_avalonia_caught_is_thrown_again_from_where_it_was_caught()
    {
        var sink = Sink();
        var original = new InvalidOperationException("The calling thread cannot access this object because a different thread owns it.");

        var thrown = Assert.Throws<AvaloniaCaughtException>(() =>
            sink.Log(LogEventLevel.Error, LogArea.Visual, this, "Exception in render loop: {Error}", original));

        Assert.Same(original, thrown.InnerException);
        Assert.StartsWith("Avalonia caught an exception (Visual): Exception in render loop:", thrown.Message);
        var entry = Assert.Single(_written);
        Assert.Equal("Avalonia error (Visual)", entry.Context);
        Assert.Contains("different thread owns it", entry.Text);
        Assert.Single(_shown);
    }

    [Fact]
    public void Errors_without_an_exception_and_warnings_are_logged_once_each()
    {
        var sink = Sink();

        sink.Log(LogEventLevel.Error, LogArea.Control, null, "Unable to load {What}", "a cursor");
        sink.Log(LogEventLevel.Error, LogArea.Control, null, "Unable to load {What}", "a cursor");
        sink.Log(LogEventLevel.Warning, LogArea.Layout, null, "Layout cycle detected");
        sink.Log(LogEventLevel.Warning, LogArea.Binding, null, "Value is null"); // bindings warn on every tab switch
        sink.Log(LogEventLevel.Information, LogArea.Visual, null, "ignored");

        Assert.Equal(
            new[] { ("Avalonia error (Control)", "Unable to load a cursor"), ("Avalonia warning (Layout)", "Layout cycle detected") },
            _written);
        Assert.Equal(new[] { "Unable to load a cursor" }, _shown); // the first error only
    }

    [Fact]
    public void A_warning_with_an_exception_is_logged_but_not_thrown()
    {
        var sink = Sink();

        sink.Log(LogEventLevel.Warning, LogArea.Control, null, "Could not load {Thing}: {Error}", "the cursor", new FormatException("bad cursor"));

        Assert.Contains("bad cursor", Assert.Single(_written).Text);
        Assert.Empty(_shown);
    }

    [Fact]
    public void A_rendering_backend_that_is_not_available_is_noted_and_nothing_else()
    {
        var sink = Sink();

        // What a machine without a usable GPU logs at start-up, before Avalonia falls back to the next backend.
        sink.Log(LogEventLevel.Error, "OpenGL", null, "Unable to initialize GLX-based rendering: {Error}", new NotSupportedException("llvmpipe is blacklisted"));
        sink.Log(LogEventLevel.Error, LogArea.Win32Platform, null, "Unable to create {What}", "a swap chain");
        sink.Log(LogEventLevel.Warning, LogArea.X11Platform, null, "SESSION_MANAGER environment variable not defined");

        Assert.Equal(3, _platform.Count);
        Assert.Contains("llvmpipe is blacklisted", _platform[0].Text);
        Assert.Empty(_written);
        Assert.Empty(_shown);
        Assert.Empty(_reported);
    }

    [Fact]
    public void Every_error_is_reported_so_tests_can_fail_on_it()
    {
        var sink = Sink();

        sink.Log(LogEventLevel.Error, LogArea.Visual, null, "Something failed");
        sink.Log(LogEventLevel.Error, LogArea.Visual, null, "Something failed"); // written once, reported each time
        sink.Log(LogEventLevel.Warning, LogArea.Visual, null, "Only a warning");

        Assert.Equal(new[] { "(Visual) Something failed", "(Visual) Something failed" }, _reported);
    }
}
