using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App;

/// <summary>Application root: dark Fluent theme, one main window.</summary>
public partial class App : Application
{
    /// <inheritdoc />
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc />
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var viewModel = new MainWindowViewModel();
            desktop.MainWindow = new MainWindow(viewModel);
            OpenStartupFile(viewModel, desktop.Args);
        }

        base.OnFrameworkInitializationCompleted();
    }

    /// <summary>
    /// `cs-run path`: a .json fixture goes to the Time/Distance tab as a demo;
    /// a telemetry file loads as Session A.
    /// </summary>
    private static void OpenStartupFile(MainWindowViewModel viewModel, string[]? args)
    {
        var path = args?.FirstOrDefault();
        if (path is null || !File.Exists(path))
        {
            return;
        }

        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            viewModel.TimeDistance.ShowFixture(Fixture.Load(path));
        }
        else
        {
            _ = viewModel.SessionA.LoadFileAsync(path);
        }
    }
}
