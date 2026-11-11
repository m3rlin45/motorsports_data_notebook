using System.ComponentModel;
using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>Hosts the histogram grid surface and the stats pane.</summary>
public partial class SuspensionView : UserControl
{
    private readonly IChartSurface _chart = ChartSurfaces.CreateHistogramGrid();
    private SuspensionViewModel? _viewModel;

    /// <summary>Create the view.</summary>
    public SuspensionView()
    {
        InitializeComponent();
        ChartHost.Content = _chart.Control;
        DataContextChanged += (_, _) => Bind(DataContext as SuspensionViewModel);
    }

    /// <summary>The chart surface.</summary>
    public IChartSurface Chart => _chart;

    private void Bind(SuspensionViewModel? vm)
    {
        if (_viewModel is not null)
        {
            _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
        }

        _viewModel = vm;
        if (vm is null)
        {
            return;
        }

        vm.PropertyChanged += OnViewModelPropertyChanged;
        if (vm.Spec is not null)
        {
            _chart.Render(vm.Spec);
        }
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(SuspensionViewModel.Spec) && _viewModel?.Spec is { } spec)
        {
            _chart.Render(spec);
        }
    }
}
