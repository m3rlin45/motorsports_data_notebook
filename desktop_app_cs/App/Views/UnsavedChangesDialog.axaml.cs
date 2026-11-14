using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>Save / Don't save / Cancel for unsaved track edits.</summary>
public partial class UnsavedChangesDialog : Window
{
    /// <summary>Create the dialog.</summary>
    public UnsavedChangesDialog()
    {
        InitializeComponent();
        DialogButtons.Arrange(Buttons);
        SaveButton.Click += (_, _) => Close(UnsavedChoice.Save);
        DiscardButton.Click += (_, _) => Close(UnsavedChoice.Discard);
        CancelButton.Click += (_, _) => Close(UnsavedChoice.Cancel);
    }

    /// <summary>
    /// Ask what to do with the unsaved edits to track <paramref name="trackName"/>.
    /// Closing the dialog any other way counts as Cancel; without a window to
    /// own it (headless use) the edits are kept by cancelling.
    /// </summary>
    public static async Task<UnsavedChoice> AskAsync(Control owner, string trackName)
    {
        if (TopLevel.GetTopLevel(owner) is not Window window)
        {
            return UnsavedChoice.Cancel;
        }

        var dialog = new UnsavedChangesDialog();
        dialog.Message.Text = $"Save changes to track “{trackName}”?";
        var choice = await dialog.ShowDialog<UnsavedChoice?>(window);
        return choice ?? UnsavedChoice.Cancel;
    }
}
