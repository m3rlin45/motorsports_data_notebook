using System.ComponentModel;
using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>Hosts the tire-grip line grid and the stats pane.</summary>
public partial class TireGripView : UserControl
{
    private readonly IChartSurface _chart = ChartSurfaces.CreateLineGrid();
    private TireGripViewModel? _viewModel;

    /// <summary>Create the view.</summary>
    public TireGripView()
    {
        InitializeComponent();
        ChartHost.Content = _chart.Control;
        DataContextChanged += (_, _) => Bind(DataContext as TireGripViewModel);
    }

    /// <summary>The chart surface.</summary>
    public IChartSurface Chart => _chart;

    private void Bind(TireGripViewModel? vm)
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
        if (e.PropertyName == nameof(TireGripViewModel.Spec) && _viewModel?.Spec is { } spec)
        {
            _chart.Render(spec);
        }
    }
}
