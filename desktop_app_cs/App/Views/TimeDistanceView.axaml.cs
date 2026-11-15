using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The Time/Distance tab: the channel picker beside the strip chart. The
/// code here is the keyboard and pointer handling that has no binding form:
/// the search box's Enter and Esc, Space on the channel lists, and dragging a
/// shown channel to another panel position.
/// </summary>
public partial class TimeDistanceView : UserControl
{
    private TimeDistanceViewModel? _viewModel;
    private int _dragIndex = -1;

    /// <summary>Create the view.</summary>
    public TimeDistanceView()
    {
        InitializeComponent();
        FitButton.Click += (_, _) => Chart.ResetView();
        SearchBox.KeyDown += OnSearchKeyDown;
        ChannelList.KeyDown += OnListKeyDown;
        ShownList.KeyDown += OnListKeyDown;
        ShownList.AddHandler(PointerPressedEvent, OnShownPointerPressed, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        ShownList.AddHandler(PointerMovedEvent, OnShownPointerMoved, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        ShownList.AddHandler(PointerReleasedEvent, (_, _) => _dragIndex = -1, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        DataContextChanged += (_, _) => Bind(DataContext as TimeDistanceViewModel);
    }

    private void Bind(TimeDistanceViewModel? vm)
    {
        if (_viewModel is not null)
        {
            _viewModel.SearchRequested -= FocusSearch;
        }

        _viewModel = vm;
        if (vm is not null)
        {
            vm.SearchRequested += FocusSearch;
        }
    }

    private void FocusSearch()
    {
        SearchBox.Focus();
        SearchBox.SelectAll();
    }

    private void OnSearchKeyDown(object? sender, KeyEventArgs e)
    {
        if (_viewModel is null)
        {
            return;
        }

        if (e.Key == Key.Enter)
        {
            _viewModel.ToggleFirstMatch();
            SearchBox.SelectAll(); // ready for the next search
            e.Handled = true;
        }
        else if (e.Key == Key.Escape && _viewModel.HasChannelFilter)
        {
            _viewModel.ClearChannelFilter();
            e.Handled = true;
        }
    }

    /// <summary>Space ticks or unticks the highlighted channel.</summary>
    private static void OnListKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Space && e.Source is not CheckBox && sender is ListBox { SelectedItem: ChannelChoiceViewModel channel })
        {
            channel.IsSelected = !channel.IsSelected;
            e.Handled = true;
        }
    }

    private int ShownIndexAt(Point position)
    {
        for (var i = 0; i < ShownList.ItemCount; i++)
        {
            if (ShownList.ContainerFromIndex(i) is { } container && container.Bounds.Contains(position))
            {
                return i;
            }
        }

        return -1;
    }

    private void OnShownPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        // A press on the checkbox itself is a tick, not the start of a drag.
        var onBox = e.Source is Visual source && Avalonia.VisualTree.VisualExtensions.FindAncestorOfType<CheckBox>(source, includeSelf: true) is { } box
            && e.GetPosition(box).X < 24;
        _dragIndex = onBox || !e.GetCurrentPoint(ShownList).Properties.IsLeftButtonPressed ? -1 : ShownIndexAt(e.GetPosition(ShownList));
    }

    private void OnShownPointerMoved(object? sender, PointerEventArgs e)
    {
        if (_dragIndex < 0 || _viewModel is null || !e.GetCurrentPoint(ShownList).Properties.IsLeftButtonPressed)
        {
            return;
        }

        var over = ShownIndexAt(e.GetPosition(ShownList));
        if (over >= 0 && over != _dragIndex)
        {
            _viewModel.MoveShown(_dragIndex, over);
            _dragIndex = over;
        }
    }
}
