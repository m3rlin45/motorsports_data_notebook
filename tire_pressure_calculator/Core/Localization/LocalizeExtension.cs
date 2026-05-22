using System;
using Avalonia.Data;
using Avalonia.Markup.Xaml;

namespace TirePressureCalculator.Localization;

/// <summary>
/// XAML markup extension: <c>{l:Localize ModeLabel}</c> binds the current
/// localized string. Used inside DataTemplates (per-corner labels);
/// direct-content labels use the explicit <c>T_*</c> properties on
/// MainViewModel instead, because compiled bindings won't walk an indexer
/// path returned from a markup extension.
/// </summary>
public sealed class LocalizeExtension : MarkupExtension
{
    public LocalizeExtension() { }

    public LocalizeExtension(string key)
    {
        Key = key;
    }

    public string Key { get; set; } = "";

    public override object ProvideValue(IServiceProvider serviceProvider)
    {
        return new Binding($"[{Key}]")
        {
            Source = Localizer.Instance,
            Mode = BindingMode.OneWay,
        };
    }
}
