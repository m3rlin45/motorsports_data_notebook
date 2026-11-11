using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Media.Imaging;
using InfernoAnalyzer.App.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;
using InfernoAnalyzer.Core.Io;
using InfernoAnalyzer.Core.Profiles;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.Tests;

/// <summary>The Sectors tab: track creation, gate editing, split table.</summary>
public class SectorsTabTests
{
    private static readonly ProfileStore BuiltinOnly = new("/nonexistent/profiles.yaml");
    private static readonly Lazy<LogFile> Sample = new(() => SessionLoader.Load(SessionParityTests.SamplePath(), BuiltinOnly));

    private static async Task<(MainWindowViewModel Window, string TracksPath, DirectoryInfo Dir)> LoadedWindow()
    {
        var dir = Directory.CreateTempSubdirectory("inferno-sectors");
        var path = Path.Combine(dir.FullName, "tracks.yaml");
        var window = new MainWindowViewModel(BuiltinOnly, new TrackStore(path));
        foreach (var tab in window.Tabs)
        {
            tab.Debounce = TimeSpan.Zero;
        }

        window.ActiveTab = window.Sectors;
        window.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
        await window.Sectors.RunNowAsync();
        return (window, path, dir);
    }

    [Fact]
    public async Task Creating_a_track_and_adding_gates_produces_a_split_table_and_saves()
    {
        var (window, path, dir) = await LoadedWindow();
        try
        {
            var tab = window.Sectors;
            Assert.Null(tab.SelectedTrack);
            Assert.Equal("Select a track or create one for this session", tab.Status);
            Assert.NotNull(tab.ReferencePath);

            tab.NewTrack();
            Assert.NotNull(tab.SelectedTrack);
            Assert.Equal("Fuji GP Sh", tab.SelectedTrack!.Name); // from the Venue metadata
            var n = tab.ReferencePath!.Value.X.Length;
            tab.AddGateAt(2 * n / 3);
            tab.AddGateAt(n / 3); // inserted before the first by lap order
            Assert.Equal(new[] { "Gate 2", "Gate 1" }, tab.Gates.Select(g => g.Name));
            await tab.RunNowAsync();

            var table = tab.Table!;
            Assert.Equal(3, table.Track.SectorCount);
            Assert.Equal(window.SessionA.SelectedLaps.Count, table.Laps.Count);
            Assert.All(table.Laps, l => Assert.True(l.Complete));
            Assert.Equal(14, table.BestLapNum);
            Assert.StartsWith($"{table.Laps.Count} laps, 3 sectors", tab.Status);

            tab.SelectedGate = tab.Gates[1];
            tab.MoveSelectedGateTo(3 * n / 4);
            Assert.Equal(new[] { "Gate 2", "Gate 1" }, tab.Gates.Select(g => g.Name)); // still in lap order
            tab.DeleteGate();
            Assert.Single(tab.Gates);
            Assert.True(tab.IsDirty);

            tab.SaveTrack();
            Assert.False(tab.IsDirty);
            var stored = new TrackStore(path).LoadAll().Single();
            Assert.Single(stored.Gates);
            Assert.Equal(tab.SelectedTrack.Id, stored.Id);

            // A fresh window for the same session finds the saved track by its anchor.
            var again = new MainWindowViewModel(BuiltinOnly, new TrackStore(path));
            again.Sectors.Debounce = TimeSpan.Zero;
            again.SessionA.Apply(Sample.Value, SessionParityTests.SamplePath());
            Assert.Equal(stored.Id, again.Sectors.SelectedTrack?.Id);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [AvaloniaFact]
    public async Task Editor_click_adds_a_gate_and_the_view_renders()
    {
        var (window, _, dir) = await LoadedWindow();
        try
        {
            window.Sectors.NewTrack();
            var view = new SectorsView { DataContext = window.Sectors };
            var top = new Window { Width = 1280, Height = 860, Content = view };
            top.Show();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();

            var editor = view.FindControl<TrackEditorControl>("Editor")!;
            var path = window.Sectors.ReferencePath!.Value;
            // Find the screen position of a path sample by probing the editor's own mapping.
            var target = path.X.Length / 2;
            Point? hit = null;
            for (var yy = 30; yy < editor.Bounds.Height - 30 && hit is null; yy += 4)
            {
                for (var xx = 30; xx < editor.Bounds.Width - 30; xx += 4)
                {
                    var near = editor.NearestSample(new Point(xx, yy));
                    if (near is { DistancePx: < 3 } && Math.Abs(near.Value.Index - target) < path.X.Length / 20)
                    {
                        hit = new Point(xx, yy);
                        break;
                    }
                }
            }

            Assert.NotNull(hit);
            editor.Press(hit!.Value);
            Assert.Single(window.Sectors.Gates);
            Assert.NotNull(window.Sectors.SelectedGate);
            await window.Sectors.RunNowAsync();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();

            var frame = top.CaptureRenderedFrame();
            Assert.NotNull(frame);
            frame!.Save(Path.Combine(RenderSmokeTests.RenderDir, "sectors.png"), new PngBitmapEncoderOptions());
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
