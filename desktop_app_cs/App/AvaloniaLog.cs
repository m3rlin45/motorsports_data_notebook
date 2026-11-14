using System.Collections.Concurrent;
using System.Globalization;
using Avalonia.Logging;

namespace InfernoAnalyzer.App;

/// <summary>
/// Sends Avalonia's own errors to the crash log. Avalonia reports what goes
/// wrong inside it through its logger and carries on: a failure while a frame
/// is drawn, for one, surfaces nowhere else, and the window simply stops
/// updating. Errors of every area are kept, and warnings of every area except
/// bindings (a binding to a value that is null for now warns on every tab
/// switch). Each distinct message is written once per run, up to a limit, so
/// an error raised on every frame cannot fill the disk; the first error of a
/// run is also shown, like an unhandled exception. <see cref="Reported"/>
/// lets the tests fail on anything logged here.
/// </summary>
internal sealed class AvaloniaLog : ILogSink
{
    private const int MaxEntries = 200;
    private readonly ConcurrentDictionary<string, bool> _seen = new();
    private int _shown;

    /// <summary>Raised for every error-level entry (on the thread that logged it), before any deduplication.</summary>
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
    public void Log(LogEventLevel level, string area, object? source, string messageTemplate, params object?[] propertyValues) =>
        Write(level, area, source, Format(messageTemplate, propertyValues));

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
            Reported?.Invoke($"({area}) {entry}");
        }

        if (_seen.Count >= MaxEntries || !_seen.TryAdd($"{level}|{area}|{entry}", true))
        {
            return;
        }

        CrashLog.WriteText($"Avalonia {level.ToString().ToLowerInvariant()} ({area})", entry);
        if (level >= LogEventLevel.Error && Interlocked.Exchange(ref _shown, 1) == 0)
        {
            // Often logged from the render thread; the dialog belongs to the UI thread.
            var firstLine = entry.Split('\n')[0].Trim();
            Avalonia.Threading.Dispatcher.UIThread.Post(() => CrashLog.Show(firstLine));
        }
    }
}
