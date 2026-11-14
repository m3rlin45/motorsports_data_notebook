using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>Session slot: open/close, file info and the lap list with its checkboxes.</summary>
public partial class SessionPanelView : UserControl
{
    /// <summary>Create the view.</summary>
    public SessionPanelView()
    {
        InitializeComponent();
        LapList.KeyDown += OnLapListKeyDown;
    }

    /// <summary>
    /// Space ticks or unticks every highlighted lap (they all take the
    /// opposite of the first one's state), so a Shift+click range or Ctrl+A
    /// followed by Space selects many laps at once.
    /// </summary>
    private void OnLapListKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Space || e.Source is CheckBox)
        {
            return;
        }

        var laps = LapList.SelectedItems?.OfType<LapItemViewModel>().ToList();
        if (laps is null || laps.Count == 0)
        {
            return;
        }

        var check = !laps[0].IsSelected;
        foreach (var lap in laps)
        {
            lap.IsSelected = check;
        }

        e.Handled = true;
    }
}
