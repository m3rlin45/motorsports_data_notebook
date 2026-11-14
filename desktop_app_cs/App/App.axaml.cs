using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using InfernoAnalyzer.App.Settings;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App;

/// <summary>Application root: Fluent theme following the OS, one main window.</summary>
public partial class App : Application
{
    /// <inheritdoc />
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc />
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            CrashLog.InstallForUiThread();
            var viewModel = new MainWindowViewModel(settings: new SettingsStore(SettingsStore.DefaultPath()));
            desktop.MainWindow = new MainWindow(viewModel);
            OpenStartupFiles(viewModel, desktop.Args);
        }

        base.OnFrameworkInitializationCompleted();
    }

    /// <summary>
    /// `cs-run path [path]`: a .json fixture goes to the Time/Distance tab as
    /// a demo; one telemetry file loads as Session A, two as A and B.
    /// </summary>
    private static void OpenStartupFiles(MainWindowViewModel viewModel, string[]? args)
    {
        var paths = (args ?? Array.Empty<string>()).Where(File.Exists).ToList();
        if (paths.Count == 0)
        {
            return;
        }

        if (paths[0].EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            viewModel.ActiveTab = viewModel.TimeDistance;
            viewModel.TimeDistance.ShowFixture(Fixture.Load(paths[0]));
        }
        else
        {
            _ = viewModel.OpenFilesAsync(paths);
        }
    }
}
