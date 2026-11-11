using System.ComponentModel;
using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Views;

/// <summary>Hosts the summary box plots, the corner detail overlay and the track map.</summary>
public partial class DriverView : UserControl
{
    private readonly IChartSurface _summary = ChartSurfaces.CreateBoxPlot();
    private readonly IChartSurface _detail = ChartSurfaces.CreateStripChart();
    private readonly IChartSurface _map = ChartSurfaces.CreateTrackMap();
    private DriverViewModel? _viewModel;

    /// <summary>Create the view.</summary>
    public DriverView()
    {
        InitializeComponent();
        SummaryHost.Content = _summary.Control;
        DetailHost.Content = _detail.Control;
        MapHost.Content = _map.Control;
        DataContextChanged += (_, _) => Bind(DataContext as DriverViewModel);
    }

    /// <summary>The box-plot surface.</summary>
    public IChartSurface SummaryChart => _summary;

    /// <summary>The detail strip-chart surface.</summary>
    public IChartSurface DetailChart => _detail;

    /// <summary>The track-map surface.</summary>
    public IChartSurface MapChart => _map;

    private void Bind(DriverViewModel? vm)
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
        RenderAll();
    }

    private void RenderAll()
    {
        if (_viewModel?.SummarySpec is { } s) _summary.Render(s);
        if (_viewModel?.DetailSpec is { } d) _detail.Render(d);
        if (_viewModel?.MapSpec is { } m) _map.Render(m);
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(DriverViewModel.SummarySpec) when _viewModel?.SummarySpec is { } s:
                _summary.Render(s);
                break;
            case nameof(DriverViewModel.DetailSpec) when _viewModel?.DetailSpec is { } d:
                _detail.Render(d);
                break;
            case nameof(DriverViewModel.MapSpec) when _viewModel?.MapSpec is { } m:
                _map.Render(m);
                break;
        }
    }
}
