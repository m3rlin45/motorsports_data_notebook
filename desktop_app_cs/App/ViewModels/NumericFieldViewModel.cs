using System.Collections;
using System.ComponentModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>
/// A number typed into a text box. The text is always what the user typed;
/// <see cref="Value"/> is the number it parses to, or null while it does not
/// parse or is out of range, in which case the box shows the validation
/// error instead of the analysis quietly running on a default.
/// </summary>
public sealed class NumericFieldViewModel : ObservableObject, INotifyDataErrorInfo
{
    private readonly string _format;
    private readonly double _min;
    private readonly bool _minExclusive;
    private readonly double _max;
    private readonly bool _allowEmpty;
    private readonly Action? _onChanged;
    private string _text = string.Empty;
    private string? _error;
    private bool _suppress;

    /// <summary>Create a field.</summary>
    /// <param name="name">What the number is, for the error text ("Front-left motion ratio").</param>
    /// <param name="format">Numeric format used when the field is set from a value.</param>
    /// <param name="min">Smallest accepted value.</param>
    /// <param name="max">Largest accepted value.</param>
    /// <param name="minExclusive">Whether <paramref name="min"/> itself is rejected.</param>
    /// <param name="allowEmpty">Whether an empty box is valid (its value is then null).</param>
    /// <param name="onChanged">Called when the user changes the text.</param>
    public NumericFieldViewModel(
        string name,
        string format = "0.###",
        double min = double.NegativeInfinity,
        double max = double.PositiveInfinity,
        bool minExclusive = false,
        bool allowEmpty = false,
        Action? onChanged = null)
    {
        Name = name;
        _format = format;
        _min = min;
        _max = max;
        _minExclusive = minExclusive;
        _allowEmpty = allowEmpty;
        _onChanged = onChanged;
        Validate();
    }

    /// <summary>What the number is.</summary>
    public string Name { get; }

    /// <summary>The text in the box.</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, value ?? string.Empty))
            {
                Validate();
                OnPropertyChanged(nameof(Value));
                if (!_suppress)
                {
                    _onChanged?.Invoke();
                }
            }
        }
    }

    /// <summary>The number, or null while the text is empty or invalid.</summary>
    public double? Value { get; private set; }

    /// <summary>Whether the text is acceptable (a number in range, or empty where that is allowed).</summary>
    public bool IsValid => _error is null;

    /// <summary>Why the text is not acceptable, or null.</summary>
    public string? Error => _error;

    /// <summary>Set the field from a number without raising the change callback (null empties it).</summary>
    public void Set(double? value)
    {
        _suppress = true;
        Text = value is { } v ? v.ToString(_format, CultureInfo.InvariantCulture) : string.Empty;
        _suppress = false;
    }

    /// <summary>Copy another field's text without raising the change callback.</summary>
    public void CopyFrom(NumericFieldViewModel other)
    {
        _suppress = true;
        Text = other.Text;
        _suppress = false;
    }

    private void Validate()
    {
        var text = _text.Trim();
        string? error = null;
        double? value = null;
        if (text.Length == 0)
        {
            error = _allowEmpty ? null : $"{Name} is required";
        }
        else if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) || !double.IsFinite(parsed))
        {
            // The usual cause: a decimal comma.
            error = $"{Name}: \"{text}\" is not a number (use a decimal point)";
        }
        else if (parsed < _min || parsed > _max || (_minExclusive && parsed == _min))
        {
            error = $"{Name} must be {Range()}";
        }
        else
        {
            value = parsed;
        }

        Value = value;
        if (error != _error)
        {
            _error = error;
            OnPropertyChanged(nameof(IsValid));
            OnPropertyChanged(nameof(Error));
            ErrorsChanged?.Invoke(this, new DataErrorsChangedEventArgs(nameof(Text)));
        }
    }

    private string Range()
    {
        static string F(double v) => v.ToString("0.###", CultureInfo.InvariantCulture);
        var low = double.IsNegativeInfinity(_min) ? null : (_minExclusive ? "greater than " : "at least ") + F(_min);
        var high = double.IsPositiveInfinity(_max) ? null : "at most " + F(_max);
        return string.Join(" and ", new[] { low, high }.Where(p => p is not null));
    }

    /// <inheritdoc />
    public bool HasErrors => _error is not null;

    /// <inheritdoc />
    public event EventHandler<DataErrorsChangedEventArgs>? ErrorsChanged;

    /// <inheritdoc />
    public IEnumerable GetErrors(string? propertyName) =>
        _error is not null && (propertyName is null or nameof(Text)) ? new[] { _error } : Array.Empty<string>();
}
