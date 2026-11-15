using Avalonia;
using Avalonia.Headless;

namespace InfernoAnalyzer.Tests;

/// <summary>
/// Headless Avalonia app for the UI tests (<see cref="Ui.Run(Action)"/>). Skia drawing is on
/// (not the no-op headless renderer) so captured frames are real images.
/// </summary>
public class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp()
    {
        // Avalonia logs what goes wrong inside it (a failed frame, for one) and carries on; Ui.Run fails the test on it.
        InfernoAnalyzer.App.AvaloniaLog.Install();
        return Build();
    }

    private static AppBuilder Build() => AppBuilder
        .Configure<InfernoAnalyzer.App.App>()
        .UseSkia()
        // The bundled font, so the saved frames look the same on every machine (and like the Linux build).
        .WithInterFont()
        .UseHeadless(new AvaloniaHeadlessPlatformOptions
        {
            UseHeadlessDrawing = false,
        });
}
