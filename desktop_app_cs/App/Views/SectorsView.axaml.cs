using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>Track editor beside the split table.</summary>
public partial class SectorsView : UserControl
{
    /// <summary>Create the view.</summary>
    public SectorsView()
    {
        InitializeComponent();
        SplitTable.LapClicked += lap =>
        {
            if (DataContext is SectorsViewModel vm)
            {
                vm.HighlightedLap = vm.HighlightedLap == lap ? null : lap;
            }
        };
    }
}
