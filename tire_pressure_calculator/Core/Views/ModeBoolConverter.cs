using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace TirePressureCalculator.Views;

/// <summary>
/// Converts <see cref="AppMode"/> ↔ bool for the <c>ToggleSwitch.IsChecked</c>
/// binding in <c>MainView.axaml</c>:
/// <list type="bullet">
///   <item><c>Manual</c> ↔ <c>false</c> (toggle off)</item>
///   <item><c>CircuitPrediction</c> ↔ <c>true</c> (toggle on)</item>
/// </list>
/// </summary>
public sealed class ModeBoolConverter : IValueConverter
{
    public static readonly ModeBoolConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        => value is AppMode m && m == AppMode.CircuitPrediction;

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => value is true ? AppMode.CircuitPrediction : AppMode.Manual;
}
