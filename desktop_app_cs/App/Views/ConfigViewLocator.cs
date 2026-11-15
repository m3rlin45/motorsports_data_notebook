using Avalonia.Controls;
using Avalonia.Controls.Templates;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// Content template of the vehicle profile window: the view for a tab's
/// channel configuration, matched by type without reflection.
/// </summary>
public sealed class ConfigViewLocator : IDataTemplate
{
    /// <inheritdoc />
    public bool Match(object? data) => data is ChannelConfigViewModel;

    /// <inheritdoc />
    public Control? Build(object? param) => param switch
    {
        SuspensionConfigViewModel => new SuspensionConfigView(),
        RollGradientConfigViewModel => new RollGradientConfigView(),
        DriverConfigViewModel => new DriverConfigView(),
        ChannelConfigViewModel => new ChannelEntriesView(),
        _ => null,
    };
}
