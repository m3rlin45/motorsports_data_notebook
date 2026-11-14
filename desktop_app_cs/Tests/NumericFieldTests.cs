using System.ComponentModel;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.Tests;

/// <summary>Numeric inputs: what counts as a number, and that a bad one is reported instead of replaced.</summary>
public class NumericFieldTests
{
    [Fact]
    public void A_valid_number_has_a_value_and_no_error()
    {
        var changes = 0;
        var field = new NumericFieldViewModel("Percentile", min: 0, max: 100, minExclusive: true, onChanged: () => changes++);
        field.Set(99.9);
        Assert.Equal(0, changes); // set from code: silent
        Assert.Equal("99.9", field.Text);

        field.Text = "90";

        Assert.Equal(1, changes);
        Assert.Equal(90, field.Value);
        Assert.True(field.IsValid);
        Assert.False(((INotifyDataErrorInfo)field).HasErrors);
    }

    [Theory]
    [InlineData("0,9", "use a decimal point")] // the decimal comma
    [InlineData("abc", "not a number")]
    [InlineData("0", "greater than 0")]
    [InlineData("101", "at most 100")]
    [InlineData("", "required")]
    public void A_bad_entry_has_no_value_and_says_why(string text, string expected)
    {
        var field = new NumericFieldViewModel("Percentile", min: 0, max: 100, minExclusive: true);
        var notified = new List<string?>();
        ((INotifyDataErrorInfo)field).ErrorsChanged += (_, e) => notified.Add(e.PropertyName);
        field.Set(50);
        notified.Clear();

        field.Text = text;

        Assert.Null(field.Value);
        Assert.False(field.IsValid);
        Assert.Contains(expected, field.Error);
        Assert.Contains(field.Error, ((INotifyDataErrorInfo)field).GetErrors(nameof(field.Text)).Cast<string>());
        Assert.Equal(new[] { "Text" }, notified); // the text box shows its validation state
    }

    [Fact]
    public void An_optional_field_may_be_empty()
    {
        var field = new NumericFieldViewModel("Front track width", allowEmpty: true, min: 0, minExclusive: true);
        Assert.True(field.IsValid);
        Assert.Null(field.Value);
        field.Text = "1520";
        Assert.Equal(1520, field.Value);
        field.Set(null);
        Assert.Equal(string.Empty, field.Text);
        Assert.True(field.IsValid);
    }
}
