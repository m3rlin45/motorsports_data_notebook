using System.ComponentModel;
using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>Hosts the roll-gradient figure (track map + scatters) and the stats pane.</summary>
public partial class RollGradientView : UserControl
{
    private readonly IChartSurface _chart = ChartSurfaces.CreateScatterGrid();
    private RollGradientViewModel? _viewModel;

    /// <summary>Create the view.</summary>
    public RollGradientView()
    {
        InitializeComponent();
        ChartHost.Content = _chart.Control;
        DataContextChanged += (_, _) => Bind(DataContext as RollGradientViewModel);
    }

    /// <summary>The chart surface.</summary>
    public IChartSurface Chart => _chart;

    private void Bind(RollGradientViewModel? vm)
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
        if (e.PropertyName == nameof(RollGradientViewModel.Spec) && _viewModel?.Spec is { } spec)
        {
            _chart.Render(spec);
        }
    }
}
