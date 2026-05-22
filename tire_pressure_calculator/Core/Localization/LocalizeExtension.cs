using System;
using Avalonia.Data;
using Avalonia.Markup.Xaml;

namespace TirePressureCalculator.Localization;

/// <summary>
/// XAML markup extension: <c>{l:Localize ModeLabel}</c> binds to
/// <see cref="Localizer.Instance"/>'s indexer. When the active language
/// changes, the binding refreshes in place via the "Item[]" notification.
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
