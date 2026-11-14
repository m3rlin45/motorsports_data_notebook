using Avalonia;

namespace InfernoAnalyzer.App;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        CrashLog.Install();
        try
        {
            // The assemblies that import the graphics libraries: SkiaSharp, HarfBuzzSharp and, for ANGLE, Avalonia.Win32.
            NativeBundle.Load(typeof(SkiaSharp.SKObject).Assembly, typeof(HarfBuzzSharp.Blob).Assembly, typeof(Win32PlatformOptions).Assembly);
            BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
        }
        catch (Exception e)
        {
            CrashLog.Write(e, "Fatal exception");
            throw;
        }
    }

    /// <summary>Also used by the Avalonia designer.</summary>
    public static AppBuilder BuildAvaloniaApp()
    {
        var builder = AppBuilder.Configure<App>().UsePlatformDetect().LogToTrace();
        // Windows and macOS have a UI font worth using (Segoe UI, San Francisco); Linux gets the bundled Inter.
        return AppFonts.UsesBundledInter ? builder.WithInterFont() : builder;
    }
}
