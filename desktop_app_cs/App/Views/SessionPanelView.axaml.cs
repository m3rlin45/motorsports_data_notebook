using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>Session slot: drop zone / file picker, file info and lap checkboxes.</summary>
public partial class SessionPanelView : UserControl
{
    private static readonly string[] Extensions = { ".xrk", ".xrz", ".ibt" };

    /// <summary>Create the view.</summary>
    public SessionPanelView()
    {
        InitializeComponent();
        DropZone.AddHandler(DragDrop.DragOverEvent, OnDragOver);
        DropZone.AddHandler(DragDrop.DropEvent, OnDrop);
        DropZone.PointerPressed += OnBrowse;
    }

    private SessionViewModel? ViewModel => DataContext as SessionViewModel;

    private static string? FirstTelemetryFile(DragEventArgs e) =>
        e.DataTransfer?.TryGetFiles()?.Select(f => f.TryGetLocalPath()).FirstOrDefault(p => p is not null && Extensions.Contains(Path.GetExtension(p).ToLowerInvariant()));

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = FirstTelemetryFile(e) is null ? DragDropEffects.None : DragDropEffects.Copy;
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        var path = FirstTelemetryFile(e);
        if (path is not null && ViewModel is { } vm)
        {
            await vm.LoadFileAsync(path);
        }
    }

    private async void OnBrowse(object? sender, PointerPressedEventArgs e)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top is null || ViewModel is not { } vm)
        {
            return;
        }

        var files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select telemetry file",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Telemetry Files") { Patterns = new[] { "*.xrk", "*.xrz", "*.ibt" } },
                new FilePickerFileType("AIM Telemetry") { Patterns = new[] { "*.xrk", "*.xrz" } },
                new FilePickerFileType("iRacing Telemetry") { Patterns = new[] { "*.ibt" } },
                FilePickerFileTypes.All,
            },
        });
        var path = files.FirstOrDefault()?.TryGetLocalPath();
        if (path is not null)
        {
            await vm.LoadFileAsync(path);
        }
    }
}
