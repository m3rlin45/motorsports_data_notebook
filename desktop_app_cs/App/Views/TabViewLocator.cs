using Avalonia.Controls;
using Avalonia.Controls.Templates;
using InfernoAnalyzer.App.Controls;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The tab area's content template: one view per tab view model, created the
/// first time the tab is shown and kept for the life of the window. Switching
/// tabs therefore swaps a finished view in and out instead of rebuilding its
/// plots, and a tab's zoom and cursor are still there when the user comes
/// back to it. The view model type is matched explicitly (no reflection), so
/// the lookup is trimming- and AOT-safe.
/// </summary>
public sealed class TabViewLocator : IDataTemplate
{
    private readonly Dictionary<AnalysisTabViewModel, Control> _views = new();

    /// <inheritdoc />
    public bool Match(object? data) => data is AnalysisTabViewModel;

    /// <inheritdoc />
    public Control? Build(object? param)
    {
        if (param is not AnalysisTabViewModel tab)
        {
            return null;
        }

        if (!_views.TryGetValue(tab, out var view))
        {
            view = Create(tab);
            view.DataContext = tab;
            _views[tab] = view;
        }

        return view;
    }

    /// <summary>The view of <paramref name="tab"/> if it has been shown, else null.</summary>
    public Control? ViewOf(AnalysisTabViewModel tab) => _views.GetValueOrDefault(tab);

    private static Control Create(AnalysisTabViewModel tab) => tab switch
    {
        TimeDistanceViewModel => new TimeDistanceView(),
        SuspensionViewModel => new SuspensionView(),
        DriverViewModel => new DriverView(),
        TireGripViewModel => new TireGripView(),
        RollGradientViewModel => new RollGradientView(),
        _ => new TextBlock { Text = tab.Header },
    };

    /// <summary>Release the chart surfaces of every view built so far (when the window closes).</summary>
    public void DisposeViews()
    {
        foreach (var view in _views.Values)
        {
            foreach (var host in Avalonia.LogicalTree.LogicalExtensions.GetLogicalDescendants(view).OfType<ChartHost>().ToList())
            {
                host.Dispose();
            }
        }

        _views.Clear();
    }
}
