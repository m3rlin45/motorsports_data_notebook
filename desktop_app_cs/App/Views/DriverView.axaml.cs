using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The Driver tab: summary box plots, the corner detail overlay and the
/// track map. Clicking a corner's column in the summary, its apex on the map
/// or its row in the list opens the detail of that corner.
/// </summary>
public partial class DriverView : UserControl
{
    /// <summary>Create the view.</summary>
    public DriverView()
    {
        InitializeComponent();
        SummaryChart.ChartClicked += OpenClickedCorner;
        MapChart.ChartClicked += OpenClickedCorner;
        // The list's selection only changes for a different corner; a click or Enter on the selected one should open it too.
        CornerList.Tapped += (_, _) => OpenSelectedCorner();
        CornerList.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Enter)
            {
                OpenSelectedCorner();
                e.Handled = true;
            }
        };
    }

    private void OpenClickedCorner(ChartPoint point)
    {
        if (point.ItemId >= 0 && DataContext is DriverViewModel vm)
        {
            vm.OpenCorner(point.ItemId);
        }
    }

    private void OpenSelectedCorner()
    {
        if (DataContext is DriverViewModel { SelectedCorner: { } corner } vm)
        {
            vm.OpenCorner(corner.Index);
        }
    }
}
