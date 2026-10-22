using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

public class StripChartBuilderTests
{
    private static Fixture TwoLapFixture()
    {
        // Reference lap: constant 20 m/s. Second lap: constant 25 m/s, so it
        // gains 0.01 s per metre → −1 s at 100 m.
        static FixtureLap Lap(int num, double speedMs, int samples)
        {
            var time = new double[samples];
            var distance = new double[samples];
            var speed = new double[samples];
            for (var i = 0; i < samples; i++)
            {
                time[i] = i * 0.5;
                distance[i] = time[i] * speedMs;
                speed[i] = speedMs * 3.6;
            }

            return new FixtureLap
            {
                Num = num,
                LapTimeS = time[^1],
                TimeS = time,
                DistanceM = distance,
                Series = { ["Speed"] = speed },
            };
        }

        return new Fixture
        {
            Source = "synthetic",
            Timebase = "Speed",
            Channels = { "Speed" },
            Units = { ["Speed"] = "km/h" },
            Laps = { Lap(1, 20, 11), Lap(2, 25, 9) },
        };
    }

    [Fact]
    public void Build_adds_one_panel_per_channel_plus_delta_time()
    {
        var spec = StripChartBuilder.Build(TwoLapFixture(), XAxisMode.Distance);

        Assert.Equal(2, spec.Subplots.Count);
        Assert.Equal("Speed", spec.Subplots[0].Title);
        Assert.Equal(2, spec.Subplots[0].Traces.Count);
        Assert.True(spec.Subplots[1].ZeroLine);
        Assert.Single(spec.Subplots[1].Traces);
        Assert.Equal("Distance (m)", spec.XLabel);
    }

    [Fact]
    public void Delta_time_is_negative_for_the_faster_lap()
    {
        var panel = StripChartBuilder.BuildDeltaTime(TwoLapFixture(), XAxisMode.Distance);
        var trace = panel.Traces[0];
        var x = trace.X.Span;
        var y = trace.Y.Span;

        // Both laps cover ≥ 100 m, so the grid runs 0..100 in 5 m steps.
        Assert.Equal(0, x[0]);
        Assert.Equal(100, x[^1]);
        Assert.Equal(0, y[0], precision: 9);
        Assert.Equal(-1.0, y[^1], precision: 9); // 100/25 − 100/20
    }

    [Fact]
    public void Time_mode_uses_the_reference_laps_elapsed_time_as_x()
    {
        var panel = StripChartBuilder.BuildDeltaTime(TwoLapFixture(), XAxisMode.Time);
        var x = panel.Traces[0].X.Span;

        Assert.Equal(0, x[0]);
        Assert.Equal(5.0, x[^1], precision: 9); // 100 m at 20 m/s
    }

    [Theory]
    [InlineData(-1.0, 0.0)]
    [InlineData(0.0, 0.0)]
    [InlineData(0.5, 5.0)]
    [InlineData(1.0, 10.0)]
    [InlineData(2.5, 25.0)]
    [InlineData(9.0, 30.0)]
    public void Interpolate_matches_numpy_interp(double at, double expected)
    {
        double[] xs = { 0, 1, 2, 3 };
        double[] ys = { 0, 10, 20, 30 };

        var result = StripChartBuilder.Interpolate(xs, ys, new[] { at });

        Assert.Equal(expected, result[0], precision: 9);
    }

    [Fact]
    public void Readout_formats_distance_and_per_lap_values()
    {
        var text = CursorReadout.Format(TwoLapFixture(), XAxisMode.Distance, 50.0);

        Assert.Equal("50 m | Speed: 72.0 / 90.0", text);
    }
}
