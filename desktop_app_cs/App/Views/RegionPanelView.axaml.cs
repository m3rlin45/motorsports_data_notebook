using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The sidebar's region selector. The chip under the pointer is shown on the
/// map; a double click on a chip ticks it alone; a click on a corner's dot on
/// the map ticks or unticks its chip.
/// </summary>
public partial class RegionPanelView : UserControl
{
    /// <summary>Create the view.</summary>
    public RegionPanelView()
    {
        InitializeComponent();
        Map.ChartClicked += point =>
        {
            if (point.ItemId >= 0 && DataContext is RegionsViewModel { IsUsed: true } vm)
            {
                vm.Toggle(point.ItemId);
            }
        };
        foreach (var chips in new[] { CornerChips, SectorChips })
        {
            chips.AddHandler(PointerMovedEvent, OnChipPointerMoved, Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
            chips.AddHandler(DoubleTappedEvent, OnChipDoubleTapped, Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
            chips.PointerExited += (_, _) => SetHovered(-1);
        }
    }

    private static RegionItemViewModel? ItemOf(object? source) => (source as Control)?.DataContext as RegionItemViewModel;

    private void OnChipPointerMoved(object? sender, PointerEventArgs e) => SetHovered(ItemOf(e.Source)?.Index ?? -1);

    // The two clicks of a double click have toggled the chip twice already; this leaves it ticked, alone.
    private void OnChipDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (ItemOf(e.Source) is { } item && DataContext is RegionsViewModel vm)
        {
            vm.Solo(item.Index);
        }
    }

    private void SetHovered(int index)
    {
        if (DataContext is RegionsViewModel vm)
        {
            vm.HoveredRegion = index;
        }
    }
}
