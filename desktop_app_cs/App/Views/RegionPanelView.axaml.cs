using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The sidebar's region selector. The row under the pointer is shown on the
/// map; a double click on a row ticks it alone; a click on a corner's dot on
/// the map ticks or unticks its row.
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
        // What the card needs changes with its rows (the height they ask for) and with what is around
        // them (the room they are given). Reported once the layout pass that changed either is over,
        // when the bounds NaturalHeight reads are those of this layout.
        RowsContent.PropertyChanged += (_, e) =>
        {
            if (e.Property == DesiredSizeProperty)
            {
                QueueNaturalHeightChanged();
            }
        };
        Rows.SizeChanged += (_, _) => QueueNaturalHeightChanged();
        foreach (var rows in new[] { CornerRows, SectorRows })
        {
            rows.AddHandler(PointerMovedEvent, OnRowPointerMoved, Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
            rows.AddHandler(DoubleTappedEvent, OnRowDoubleTapped, Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
            rows.PointerExited += (_, _) => SetHovered(-1);
        }
    }

    /// <summary>
    /// How tall the card has to be to show every row without scrolling: what it is now, with the
    /// rows' own height in place of the room they have. 0 before the card has been laid out.
    /// </summary>
    /// <remarks>
    /// The rows' own height is what their panel asks for, not what it is given: inside the scroll
    /// viewer it is stretched to the room there is, so its bounds would say "as tall as the card".
    /// </remarks>
    public double NaturalHeight => Bounds.Height <= 0 ? 0 : Bounds.Height - Rows.Bounds.Height + RowsContent.DesiredSize.Height;

    /// <summary>Raised when <see cref="NaturalHeight"/> may have changed (rows listed or removed, a hint shown or hidden).</summary>
    public event Action? NaturalHeightChanged;

    private bool _naturalHeightQueued;

    private void QueueNaturalHeightChanged()
    {
        if (_naturalHeightQueued)
        {
            return;
        }

        _naturalHeightQueued = true;
        Avalonia.Threading.Dispatcher.UIThread.Post(
            () =>
            {
                _naturalHeightQueued = false;
                NaturalHeightChanged?.Invoke();
            },
            Avalonia.Threading.DispatcherPriority.Render);
    }

    private static RegionItemViewModel? ItemOf(object? source) => (source as Control)?.DataContext as RegionItemViewModel;

    private void OnRowPointerMoved(object? sender, PointerEventArgs e) => SetHovered(ItemOf(e.Source)?.Index ?? -1);

    // The two clicks of a double click have toggled the row twice already; this leaves it ticked, alone.
    private void OnRowDoubleTapped(object? sender, TappedEventArgs e)
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
