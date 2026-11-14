using Avalonia.Controls;
using Avalonia.Input.Platform;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>The Sectors tab: the track editor beside the split table.</summary>
public partial class SectorsView : UserControl
{
    /// <summary>Create the view.</summary>
    public SectorsView()
    {
        InitializeComponent();
        CopyTableButton.Click += async (_, _) =>
        {
            if (DataContext is SectorsViewModel vm && TopLevel.GetTopLevel(this)?.Clipboard is { } clipboard)
            {
                await clipboard.SetTextAsync(vm.TableAsTsv());
            }
        };
        DataContextChanged += (_, _) =>
        {
            if (DataContext is SectorsViewModel vm)
            {
                vm.ConfirmUnsaved = name => UnsavedChangesDialog.AskAsync(this, name);
            }
        };
    }
}
