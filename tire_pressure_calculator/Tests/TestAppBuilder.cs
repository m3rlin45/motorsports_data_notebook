using Avalonia;
using Avalonia.Headless;
using TirePressureCalculator.Tests;

[assembly: AvaloniaTestApplication(typeof(TestAppBuilder))]

namespace TirePressureCalculator.Tests;

public class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder
        .Configure<TirePressureCalculator.App>()
        .UseHeadless(new AvaloniaHeadlessPlatformOptions
        {
            UseHeadlessDrawing = true,
        });
}
