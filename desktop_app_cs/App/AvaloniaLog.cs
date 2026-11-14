using System.Collections.Concurrent;
using System.Globalization;
using Avalonia.Logging;

namespace InfernoAnalyzer.App;

/// <summary>
/// Stops Avalonia swallowing failures. Avalonia reports what goes wrong
/// inside it through its logger and carries on: an exception thrown while a
/// frame is drawn, for one, is caught in the render loop and surfaces nowhere
/// else, and the window simply stops updating. Here an error that carries an
/// exception is thrown again from the place Avalonia caught it, so it ends
/// the way any unhandled exception does: on the UI thread through the
/// dispatcher's handler, anywhere else by taking the process down with the
/// exception in the crash log (<see cref="CrashLog"/>). Errors without an
/// exception, and warnings of every area except bindings (a binding to a
/// value that is null for now warns on every tab switch), are written to the
/// crash log, each distinct message once per run up to a limit, and the
/// first error of a run is shown. <see cref="Reported"/> lets the tests fail
/// on anything logged here.
/// </summary>
internal sealed class AvaloniaLog : ILogSink
{
    private const int MaxEntries = 200;
    private readonly ConcurrentDictionary<string, bool> _seen = new();
    private readonly Action<string, string> _write;
    private readonly Action<string> _show;
    private readonly Action<string> _report;
    private int _shown;

    /// <summary>
    /// Create a sink; the defaults write to the crash log, show the error dialog on the UI thread
    /// and raise <see cref="Reported"/>.
    /// </summary>
    internal AvaloniaLog(Action<string, string>? write = null, Action<string>? show = null, Action<string>? report = null)
    {
        _write = write ?? CrashLog.WriteText;
        _show = show ?? (message => Avalonia.Threading.Dispatcher.UIThread.Post(() => CrashLog.Show(message)));
        _report = report ?? (error => Reported?.Invoke(error));
    }

    /// <summary>Raised by the installed sink for every error-level entry (on the thread that logged it), before any deduplication.</summary>
    internal static event Action<string>? Reported;

    /// <summary>Route Avalonia's logging here. Call before the app starts.</summary>
    public static void Install() => Logger.Sink = new AvaloniaLog();

    /// <inheritdoc />
    public bool IsEnabled(LogEventLevel level, string area) =>
        level >= LogEventLevel.Error || (level == LogEventLevel.Warning && area != LogArea.Binding);

    /// <inheritdoc />
    public void Log(LogEventLevel level, string area, object? source, string messageTemplate) =>
        Write(level, area, source, messageTemplate);

    /// <inheritdoc />
    public void Log(LogEventLevel level, string area, object? source, string messageTemplate, params object?[] propertyValues)
    {
        var message = Format(messageTemplate, propertyValues);
        Write(level, area, source, message);
        if (level >= LogEventLevel.Error && propertyValues.OfType<Exception>().FirstOrDefault() is { } caught)
        {
            // Avalonia caught this and is about to carry on without it.
            throw new AvaloniaCaughtException($"Avalonia caught an exception ({area}): {message.Split('\n')[0].Trim()}", caught);
        }
    }

    /// <summary>Fill the template's {Named} holes in order; values left over (an exception, usually) follow on their own lines.</summary>
    internal static string Format(string template, object?[] values)
    {
        var text = new System.Text.StringBuilder();
        var used = 0;
        for (var i = 0; i < template.Length; i++)
        {
            var close = template[i] == '{' ? template.IndexOf('}', i) : -1;
            if (close < 0 || used >= values.Length)
            {
                text.Append(template[i]);
                continue;
            }

            text.Append(Convert.ToString(values[used++], CultureInfo.InvariantCulture));
            i = close;
        }

        for (; used < values.Length; used++)
        {
            text.AppendLine().Append(Convert.ToString(values[used], CultureInfo.InvariantCulture));
        }

        return text.ToString();
    }

    private void Write(LogEventLevel level, string area, object? source, string message)
    {
        if (!IsEnabled(level, area))
        {
            return;
        }

        var entry = source is null ? message : $"{source.GetType().Name}: {message}";
        if (level >= LogEventLevel.Error)
        {
            _report($"({area}) {entry}");
        }

        if (_seen.Count >= MaxEntries || !_seen.TryAdd($"{level}|{area}|{entry}", true))
        {
            return;
        }

        _write($"Avalonia {level.ToString().ToLowerInvariant()} ({area})", entry);
        if (level >= LogEventLevel.Error && Interlocked.Exchange(ref _shown, 1) == 0)
        {
            _show(entry.Split('\n')[0].Trim());
        }
    }
}

/// <summary>An exception Avalonia caught and logged, thrown again by <see cref="AvaloniaLog"/>; the original is the inner exception.</summary>
public sealed class AvaloniaCaughtException : Exception
{
    /// <summary>Wrap <paramref name="inner"/>, the exception Avalonia caught.</summary>
    public AvaloniaCaughtException(string message, Exception inner)
        : base(message, inner)
    {
    }
}
