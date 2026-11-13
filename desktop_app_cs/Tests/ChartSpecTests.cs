using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

public class ChartSpecTests
{
    private static readonly TraceSpec Trace = new(
        "t",
        new double[] { 0, 1, 2, 3 },
        new double[] { 0, 10, 20, 30 },
        "#000000");

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(0.5, 5.0)]
    [InlineData(1.0, 10.0)]
    [InlineData(2.5, 25.0)]
    [InlineData(3.0, 30.0)]
    public void ValueAt_interpolates_between_samples(double at, double expected)
    {
        Assert.Equal(expected, Trace.ValueAt(at), precision: 9);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(3.1)]
    [InlineData(double.NaN)]
    public void ValueAt_is_nan_outside_the_trace(double at)
    {
        Assert.True(double.IsNaN(Trace.ValueAt(at)));
    }

    [Fact]
    public void ValueAt_is_nan_for_an_empty_trace()
    {
        var empty = new TraceSpec("e", ReadOnlyMemory<double>.Empty, ReadOnlyMemory<double>.Empty, "#000000");
        Assert.True(double.IsNaN(empty.ValueAt(1.0)));
    }

    [Fact]
    public void FormatX_depends_on_axis_mode()
    {
        var subplots = Array.Empty<SubplotSpec>();
        Assert.Equal("2158 m", new StripChartSpec(subplots, XAxisMode.Distance).FormatX(2157.6));
        Assert.Equal("61.25 s", new StripChartSpec(subplots, XAxisMode.Time).FormatX(61.249));
    }
}
