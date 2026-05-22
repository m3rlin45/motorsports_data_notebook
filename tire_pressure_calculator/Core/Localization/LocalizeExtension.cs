using System;
using Avalonia.Data;
using Avalonia.Markup.Xaml;
using Avalonia.Markup.Xaml.MarkupExtensions;

namespace TirePressureCalculator.Localization;

/// <summary>
/// XAML markup extension: <c>{l:Localize ModeLabel}</c> resolves to the
/// current localized string. When the active language changes the producer
/// pushes the new value through a one-way binding so the UI refreshes
/// without a XAML reload.
///
/// We return a <see cref="ReflectionBindingExtension"/>-backed binding
/// rather than a hand-rolled <see cref="Binding"/> because compiled
/// bindings (enabled project-wide via
/// <c>AvaloniaUseCompiledBindingsByDefault=true</c>) refuse to evaluate a
/// raw <c>Binding</c> returned from a markup extension when the host
/// element's <c>x:DataType</c> doesn't expose the requested path —
/// indexer paths on <see cref="Localizer.Instance"/> fall in that bucket.
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
        var binding = new ReflectionBindingExtension($"[{Key}]")
        {
            Source = Localizer.Instance,
            Mode = BindingMode.OneWay,
        };
        return binding.ProvideValue(serviceProvider);
    }
}
