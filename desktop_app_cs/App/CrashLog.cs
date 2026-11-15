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
public static partial class CrashLog
{
    /// <summary>Where unhandled exceptions are logged.</summary>
    public static string LogPath => Path.Combine(Path.GetDirectoryName(SettingsStore.DefaultPath())!, "analyzer_crash.log");

    /// <summary>Hook the process-wide handlers. Call once, before the app starts.</summary>
    public static void Install()
    {
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            var exception = e.ExceptionObject as Exception;
            Write(exception, "Unhandled exception");
            if (e.IsTerminating)
            {
                // Off the UI thread there is no carrying on: say why the window is about to vanish.
                ShowFatal($"Inferno Analyzer has to close.\n\n{exception?.GetBaseException().Message}\n\nDetails were written to:\n{LogPath}");
            }
        };
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
            Show(e.Exception.Message);
        };
    }

    /// <summary>Append <paramref name="exception"/> to the log; never throws.</summary>
    public static void Write(Exception? exception, string context) =>
        WriteText(context, exception?.ToString() ?? "(no exception object)");

    /// <summary>Append an entry headed <paramref name="context"/> to the log; never throws.</summary>
    public static void WriteText(string context, string text) => WriteText(context, text, LogPath);

    /// <summary>Append an entry headed <paramref name="context"/> to the log file at <paramref name="path"/>; never throws.</summary>
    internal static void WriteText(string context, string text, string path)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var entry = new StringBuilder()
                .AppendLine($"=== {DateTime.Now:yyyy-MM-dd HH:mm:ss} {context} (version {typeof(CrashLog).Assembly.GetName().Version}) ===")
                .AppendLine(text.TrimEnd())
                .AppendLine();
            File.AppendAllText(path, entry.ToString());
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Nothing more can be done; failing to log must not hide the original problem behind a second one.
        }
    }

    /// <summary>A message box that needs nothing of Avalonia (the render thread may be the one that died). Windows only; elsewhere the log is the report.</summary>
    private static void ShowFatal(string text)
    {
        if (OperatingSystem.IsWindows())
        {
            _ = MessageBoxW(0, text, "Inferno Analyzer", 0x10 | 0x2000 | 0x10000); // MB_ICONERROR | MB_TASKMODAL | MB_SETFOREGROUND
        }
        else
        {
            Console.Error.WriteLine(text);
        }
    }

    [System.Runtime.InteropServices.LibraryImport("user32", StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf16)]
    private static partial int MessageBoxW(nint owner, string text, string caption, uint type);

    /// <summary>Tell the user something failed and where the details are; the app carries on. UI thread only.</summary>
    internal static void Show(string message)
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
                    new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap, FontWeight = FontWeight.SemiBold },
                    new TextBlock { Text = $"The app will keep running. Details were written to:\n{LogPath}", TextWrapping = TextWrapping.Wrap },
                    close,
                },
            },
        };
        close.Click += (_, _) => dialog.Close();
        dialog.ShowDialog(owner);
    }
}
