using Avalonia;
using Avalonia.Browser;

namespace TirePressureCalculator;

internal sealed class Program
{
    private static Task Main(string[] args) =>
        BuildAvaloniaApp().StartBrowserAppAsync("out");

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>();
}
