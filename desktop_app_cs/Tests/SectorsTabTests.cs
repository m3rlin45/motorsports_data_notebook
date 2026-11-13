using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
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
    public Task New_track_is_named_from_the_session_metadata() => Ui.Run(() =>
    {
        Assert.Equal("Fuji GP Sh", SectorsViewModel.TrackNameFor(new Dictionary<string, string> { ["Venue"] = "Fuji GP Sh" }));
        Assert.Equal(
            "Phillip Island Circuit",
            SectorsViewModel.TrackNameFor(new Dictionary<string, string> { ["track_name"] = "phillipisland", ["track_display_name"] = "Phillip Island Circuit" }));
        Assert.Equal("New track", SectorsViewModel.TrackNameFor(new Dictionary<string, string> { ["Venue"] = string.Empty }));
    });

    [Fact]
    public Task Creating_a_track_and_adding_gates_produces_a_split_table_and_saves() => Ui.Run(async () =>
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
    });

    [Fact]
    public Task Editor_click_adds_a_gate_and_the_view_renders() => Ui.Run(async () =>
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
            RenderSmokeTests.SaveRender(frame!, "sectors.png");
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });

    [Fact]
    public Task Renaming_a_track_keeps_its_unsaved_gates_and_saves_under_the_new_name() => Ui.Run(async () =>
    {
        var (window, path, dir) = await LoadedWindow();
        try
        {
            var tab = window.Sectors;
            tab.NewTrack();
            var n = tab.ReferencePath!.Value.X.Length;
            tab.AddGateAt(n / 2);
            var id = tab.SelectedTrack!.Id;

            // Typing in the name box, one keystroke at a time.
            tab.TrackName = "Fuji GP S";
            tab.TrackName = "Fuji GP Short";

            Assert.Single(tab.Gates); // the gate just placed is still there
            Assert.True(tab.IsDirty);
            Assert.True(tab.SaveTrackCommand.CanExecute(null));
            Assert.Equal(id, tab.SelectedTrack!.Id);

            tab.SaveTrack(); // must not throw: the track is found by id, not by instance

            var stored = new TrackStore(path).LoadAll().Single();
            Assert.Equal("Fuji GP Short", stored.Name);
            Assert.Equal(id, stored.Id);
            Assert.Single(stored.Gates);
            Assert.Single(tab.Tracks);
            Assert.Equal("Fuji GP Short", tab.Tracks[0].Name); // the picker shows the new name
            Assert.Same(tab.Tracks[0], tab.SelectedTrack);
            Assert.False(tab.IsDirty);
            Assert.False(tab.SaveTrackCommand.CanExecute(null));

            // A rename alone is a change worth saving; an emptied name keeps the old one.
            tab.TrackName = "  ";
            Assert.True(tab.IsDirty);
            Assert.Equal("Fuji GP Short", tab.CurrentTrack().Name);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });

    [Fact]
    public Task Gate_edits_can_be_undone_one_step_at_a_time() => Ui.Run(async () =>
    {
        var (window, _, dir) = await LoadedWindow();
        try
        {
            var tab = window.Sectors;
            tab.NewTrack();
            Assert.False(tab.UndoCommand.CanExecute(null));
            Assert.False(tab.DeleteGateCommand.CanExecute(null)); // nothing selected

            var n = tab.ReferencePath!.Value.X.Length;
            tab.AddGateAt(n / 3);
            tab.AddGateAt(2 * n / 3);
            var second = tab.NearestIndex(tab.Gates[1]);

            // A nudge is one step; a drag is one step however many pointer moves it takes.
            tab.SelectedGate = tab.Gates[1];
            tab.NudgeSelectedGate(-10);
            Assert.Equal(second - 10, tab.NearestIndex(tab.Gates[1]));
            tab.BeginGateMove();
            tab.MoveSelectedGateTo(second - 40);
            tab.MoveSelectedGateTo(second - 80);

            tab.Undo();
            Assert.Equal(second - 10, tab.NearestIndex(tab.Gates[1]));
            tab.Undo();
            Assert.Equal(second, tab.NearestIndex(tab.Gates[1]));

            tab.SelectedGate = tab.Gates[0];
            Assert.True(tab.DeleteGateCommand.CanExecute(null));
            tab.SelectedGate.Name = "Hairpin";
            tab.DeleteGate();
            Assert.Single(tab.Gates);
            tab.Undo();
            Assert.Equal(new[] { "Hairpin", "Gate 2" }, tab.Gates.Select(g => g.Name));

            tab.Undo();
            tab.Undo();
            Assert.Empty(tab.Gates);
            Assert.False(tab.UndoCommand.CanExecute(null));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });

    [Fact]
    public Task Switching_tracks_with_unsaved_edits_asks_first() => Ui.Run(async () =>
    {
        var (window, path, dir) = await LoadedWindow();
        try
        {
            var tab = window.Sectors;
            tab.NewTrack();
            var first = tab.SelectedTrack!;
            tab.SaveTrack();
            tab.NewTrack();
            var second = tab.SelectedTrack!;
            tab.SaveTrack();
            var n = tab.ReferencePath!.Value.X.Length;
            tab.AddGateAt(n / 2);

            var asked = new List<string>();
            var answer = UnsavedChoice.Cancel;
            tab.ConfirmUnsaved = name =>
            {
                asked.Add(name);
                return Task.FromResult(answer);
            };

            // Cancel: stay on the edited track with its gate.
            tab.SelectedTrack = tab.Tracks.First(t => t.Id == first.Id);
            Assert.Equal(new[] { second.Name }, asked);
            Assert.Equal(second.Id, tab.SelectedTrack!.Id);
            Assert.Single(tab.Gates);
            Assert.True(tab.IsDirty);

            // Save: the gate is stored, then the other track is shown.
            answer = UnsavedChoice.Save;
            tab.SelectedTrack = tab.Tracks.First(t => t.Id == first.Id);
            Assert.Equal(first.Id, tab.SelectedTrack!.Id);
            Assert.Empty(tab.Gates);
            Assert.False(tab.IsDirty);
            Assert.Single(new TrackStore(path).LoadAll().First(t => t.Id == second.Id).Gates);

            // Don't save: the edit is dropped.
            tab.AddGateAt(n / 4);
            answer = UnsavedChoice.Discard;
            tab.SelectedTrack = tab.Tracks.First(t => t.Id == second.Id);
            Assert.Equal(second.Id, tab.SelectedTrack!.Id);
            Assert.Single(tab.Gates);
            Assert.Empty(new TrackStore(path).LoadAll().First(t => t.Id == first.Id).Gates);
            Assert.Equal(3, asked.Count);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });

    [Fact]
    public Task Split_table_copies_as_tsv_and_the_keyboard_moves_the_highlight() => Ui.Run(async () =>
    {
        var (window, _, dir) = await LoadedWindow();
        try
        {
            var tab = window.Sectors;
            tab.NewTrack();
            tab.AddGateAt(tab.ReferencePath!.Value.X.Length / 2);
            await tab.RunNowAsync();

            var lines = tab.TableAsTsv().Split('\n');
            Assert.Equal("Lap\tS1\tS2\tLap time", lines[0]);
            Assert.Equal(tab.Table!.Laps.Count + 2, lines.Length);
            Assert.StartsWith("Theoretical best\t", lines[^1]);
            Assert.EndsWith("\t2:05.056", lines.First(l => l.StartsWith("14\t")));

            var view = new SectorsView { DataContext = tab };
            var top = new Window { Width = 1280, Height = 860, Content = view };
            top.Show();
            Avalonia.Threading.Dispatcher.UIThread.RunJobs();
            var table = view.FindControl<SplitTableControl>("SplitTable")!;
            Assert.True(table.Focusable);
            table.Focus();

            top.KeyPressQwerty(Avalonia.Input.PhysicalKey.ArrowDown, Avalonia.Input.RawInputModifiers.None);
            Assert.Equal(tab.Table.Laps[0].LapNum, tab.HighlightedLap); // two-way: the view model follows the control
            top.KeyPressQwerty(Avalonia.Input.PhysicalKey.ArrowDown, Avalonia.Input.RawInputModifiers.None);
            Assert.Equal(tab.Table.Laps[1].LapNum, tab.HighlightedLap);
            top.KeyPressQwerty(Avalonia.Input.PhysicalKey.Escape, Avalonia.Input.RawInputModifiers.None);
            Assert.Null(tab.HighlightedLap);

            // The editor's keys act on the selected gate.
            var editor = view.FindControl<TrackEditorControl>("Editor")!;
            tab.SelectedGate = tab.Gates[0];
            var at = tab.NearestIndex(tab.Gates[0]);
            editor.Focus();
            top.KeyPressQwerty(Avalonia.Input.PhysicalKey.ArrowRight, Avalonia.Input.RawInputModifiers.Shift);
            Assert.Equal(at + 10, tab.NearestIndex(tab.Gates[0]));
            top.KeyPressQwerty(Avalonia.Input.PhysicalKey.Delete, Avalonia.Input.RawInputModifiers.None);
            Assert.Empty(tab.Gates);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    });
}
