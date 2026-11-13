using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.Tests;

/// <summary>The channel search box filters the list and its clear button empties it.</summary>
public class ChannelSearchRenderTest
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");

    [Fact]
    public Task Search_box_filters_and_clears() => Ui.Run(async () =>
    {
        var window = new MainWindowViewModel(BuiltinOnly);
        window.TimeDistance.Debounce = TimeSpan.Zero;
        window.SessionA.Apply(SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly), SessionParityTests.SamplePath());
        await window.TimeDistance.RunNowAsync();

        var view = new TimeDistanceView { DataContext = window.TimeDistance };
        var top = new Window { Width = 1280, Height = 860, Content = view };
        top.Show();
        window.TimeDistance.ChannelFilter = "shock";
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();

        var frame = top.CaptureRenderedFrame();
        Assert.NotNull(frame);
        RenderSmokeTests.SaveRender(frame!, "channel_search.png");

        var clear = view.FindControl<Button>("ClearFilterButton")!;
        Assert.True(clear.IsVisible);
        clear.Command!.Execute(null);
        Avalonia.Threading.Dispatcher.UIThread.RunJobs();
        Assert.Equal(string.Empty, window.TimeDistance.ChannelFilter);
        Assert.All(window.TimeDistance.Channels, c => Assert.True(c.IsMatch));
        Assert.False(clear.IsVisible); // hidden while the box is empty
    });
}
