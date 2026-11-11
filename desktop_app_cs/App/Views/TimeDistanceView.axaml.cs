using System.ComponentModel;
using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>Hosts the strip-chart surface and renders the view model's spec.</summary>
public partial class TimeDistanceView : UserControl
{
    private readonly IChartSurface _chart = ChartSurfaces.CreateStripChart();
    private TimeDistanceViewModel? _viewModel;

    /// <summary>Create the view.</summary>
    public TimeDistanceView()
    {
        InitializeComponent();
        ChartHost.Content = _chart.Control;
        DataContextChanged += (_, _) => Bind(DataContext as TimeDistanceViewModel);
    }

    /// <summary>The chart surface (tests drive it directly).</summary>
    public IChartSurface Chart => _chart;

    private void Bind(TimeDistanceViewModel? vm)
    {
        if (_viewModel is not null)
        {
            _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
            _viewModel.Cursors.Remove(_chart);
        }

        _viewModel = vm;
        if (vm is null)
        {
            return;
        }

        vm.Cursors.Add(_chart);
        vm.PropertyChanged += OnViewModelPropertyChanged;
        if (vm.Spec is not null)
        {
            _chart.Render(vm.Spec);
        }
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(TimeDistanceViewModel.Spec) && _viewModel?.Spec is { } spec)
        {
            _chart.Render(spec);
        }
    }
}
