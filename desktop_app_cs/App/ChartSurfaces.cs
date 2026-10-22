using InfernoAnalyzer.Charting;
using InfernoAnalyzer.Charting.ScottPlot;

namespace InfernoAnalyzer.App;

/// <summary>
/// The one place the chart backend is chosen. Swapping libraries means a new
/// adapter project and a new line here; views and viewmodels never change.
/// </summary>
public static class ChartSurfaces
{
    /// <summary>Create a surface able to render <see cref="StripChartSpec"/>.</summary>
    public static IChartSurface CreateStripChart() => new ScottPlotStripChart();
}
