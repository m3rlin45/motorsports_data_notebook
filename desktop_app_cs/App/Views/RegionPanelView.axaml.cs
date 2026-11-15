using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The region list of the statistical tabs. The row under the pointer is
/// shown on the map; a click on a corner's dot on the map ticks or unticks
/// its row.
/// </summary>
public partial class RegionPanelView : UserControl
{
    /// <summary>Create the view.</summary>
    public RegionPanelView()
    {
        InitializeComponent();
        Map.ChartClicked += point =>
        {
            if (point.ItemId >= 0 && DataContext is RegionsViewModel vm)
            {
                vm.Toggle(point.ItemId);
            }
        };
        foreach (var rows in new[] { CornerRows, SectorRows })
        {
            rows.AddHandler(PointerMovedEvent, OnRowPointerMoved, Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
            rows.PointerExited += (_, _) => SetHovered(-1);
        }
    }

    private void OnRowPointerMoved(object? sender, PointerEventArgs e) =>
        SetHovered((e.Source as Control)?.DataContext is RegionItemViewModel item ? item.Index : -1);

    private void SetHovered(int index)
    {
        if (DataContext is RegionsViewModel vm)
        {
            vm.HoveredRegion = index;
        }
    }
}
