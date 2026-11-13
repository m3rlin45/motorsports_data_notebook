using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>Renders the Time/Distance view with Session A and B overlaid.</summary>
public class TimeDistanceCompareRenderTest
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");

    [Fact]
    public Task Two_sessions_render_in_one_strip_chart() => Ui.Run(async () =>
    {
        var sample = SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly);
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        window.SessionA.Apply(sample, SessionParityTests.SamplePath());
        window.SessionB.Apply(sample, SessionParityTests.SamplePath());
        foreach (var lap in window.SessionB.Laps)
        {
            lap.IsSelected = lap.Num == 12;
        }

        await window.TimeDistance.RunNowAsync();

        var view = new TimeDistanceView { DataContext = window.TimeDistance };
        var top = new Window { Width = 1280, Height = 860, Content = view };
        top.Show();
        window.TimeDistance.Cursors.SetCursor(2000);
        var frame = top.CaptureRenderedFrame();
        Assert.NotNull(frame);
        RenderSmokeTests.SaveRender(frame!, "strip_chart_compare.png");
    });
}
