using Avalonia;

namespace InfernoAnalyzer.App;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    /// <summary>Also used by the Avalonia designer.</summary>
    public static AppBuilder BuildAvaloniaApp()
    {
        var builder = AppBuilder.Configure<App>().UsePlatformDetect().LogToTrace();
        // Windows and macOS have a UI font worth using (Segoe UI, San Francisco); Linux gets the bundled Inter.
        return AppFonts.UsesBundledInter ? builder.WithInterFont() : builder;
    }
}
