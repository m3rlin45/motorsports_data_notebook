using Avalonia.Controls;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// Button order of a dialog's button row. Dialogs declare their buttons in
/// the Windows order (the affirmative action first, Cancel last); on macOS
/// and Linux desktops the affirmative action goes on the far right instead.
/// </summary>
public static class DialogButtons
{
    /// <summary>
    /// Reorder <paramref name="row"/> for the platform: unchanged on Windows,
    /// elsewhere the first child (the affirmative action) moves to the end.
    /// </summary>
    public static void Arrange(Panel row)
    {
        if (OperatingSystem.IsWindows() || row.Children.Count < 2)
        {
            return;
        }

        var affirmative = row.Children[0];
        row.Children.RemoveAt(0);
        row.Children.Add(affirmative);
    }
}
