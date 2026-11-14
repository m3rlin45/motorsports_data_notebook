using System.Text;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using InfernoAnalyzer.App.Settings;

namespace InfernoAnalyzer.App;

/// <summary>
/// Last-chance exception handling. The app is a windowed executable with no
/// console, so an exception that escapes an event handler would otherwise end
/// the process without a trace. Unhandled exceptions are appended to a log
/// beside the settings file, and those on the UI thread are also shown, after
/// which the app carries on: a failed analysis or a bad file should not cost
/// the user the session they have open. A frozen UI thread is reported
/// here as well (<see cref="HangWatch"/>).
/// </summary>
public static class CrashLog
{
    /// <summary>Where unhandled exceptions are logged.</summary>
    public static string LogPath => Path.Combine(Path.GetDirectoryName(SettingsStore.DefaultPath())!, "analyzer_crash.log");

    /// <summary>Hook the process-wide handlers. Call once, before the app starts.</summary>
    public static void Install()
    {
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Write(e.ExceptionObject as Exception, "Unhandled exception");
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Write(e.Exception, "Unobserved task exception");
            e.SetObserved();
        };
    }

    /// <summary>Hook the UI thread's dispatcher. Call once the app has started.</summary>
    public static void InstallForUiThread()
    {
        Dispatcher.UIThread.UnhandledException += (_, e) =>
        {
            Write(e.Exception, "Unhandled exception on the UI thread");
            e.Handled = true;
            Show(e.Exception);
        };
    }

    /// <summary>Append <paramref name="exception"/> to the log; never throws.</summary>
    public static void Write(Exception? exception, string context) =>
        WriteText(context, exception?.ToString() ?? "(no exception object)");

    /// <summary>Append an entry headed <paramref name="context"/> to the log; never throws.</summary>
    public static void WriteText(string context, string text)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
            var entry = new StringBuilder()
                .AppendLine($"=== {DateTime.Now:yyyy-MM-dd HH:mm:ss} {context} (version {typeof(CrashLog).Assembly.GetName().Version}) ===")
                .AppendLine(text.TrimEnd())
                .AppendLine();
            File.AppendAllText(LogPath, entry.ToString());
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Nothing more can be done; failing to log must not hide the original problem behind a second one.
        }
    }

    private static void Show(Exception exception)
    {
        if (Avalonia.Application.Current?.ApplicationLifetime is not Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime { MainWindow: { IsVisible: true } owner })
        {
            return;
        }

        var close = new Button { Content = "Close", IsDefault = true, IsCancel = true, HorizontalAlignment = HorizontalAlignment.Right, MinWidth = 88 };
        var dialog = new Window
        {
            Title = "Something went wrong",
            SizeToContent = SizeToContent.WidthAndHeight,
            CanResize = false,
            ShowInTaskbar = false,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Content = new StackPanel
            {
                Margin = new Avalonia.Thickness(24),
                Spacing = 12,
                MaxWidth = 480,
                Children =
                {
                    new TextBlock { Text = exception.Message, TextWrapping = TextWrapping.Wrap, FontWeight = FontWeight.SemiBold },
                    new TextBlock { Text = $"The app will keep running. Details were written to:\n{LogPath}", TextWrapping = TextWrapping.Wrap },
                    close,
                },
            },
        };
        close.Click += (_, _) => dialog.Close();
        dialog.ShowDialog(owner);
    }
}
