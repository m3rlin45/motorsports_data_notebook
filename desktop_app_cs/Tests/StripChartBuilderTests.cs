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

    /// <summary>Lap 2 of the two-lap fixture with lap 1 before it and a lap 3 after it.</summary>
    private static FixtureLap LapWithNeighbours()
    {
        var fixture = TwoLapFixture();
        var lap = fixture.Laps[1]; // 25 m/s, 9 samples, 100 m, 4 s
        lap.Before = fixture.Laps[0]; // 20 m/s, 11 samples, 100 m, 5 s
        lap.After = new FixtureLap
        {
            Num = 3,
            LapTimeS = 2,
            TimeS = new double[] { 0, 1, 2 },
            DistanceM = new double[] { 0, 30, 60 },
            Series = { ["Speed"] = new double[] { 108, 108, 108 } },
        };
        return lap;
    }

    [Fact]
    public void Stitch_places_the_neighbours_either_side_of_the_lap_by_distance()
    {
        var stitched = StripChartBuilder.Stitch(LapWithNeighbours(), XAxisMode.Distance);

        Assert.Equal(11 + 9 + 3, stitched.X.Length);
        Assert.Equal(-100, stitched.X[0]); // lap 1 starts one lap length before the start line
        Assert.Equal(0, stitched.X[11]); // lap 2 starts at the start line
        Assert.Equal(100 + 60, stitched.X[^1]); // lap 3 continues after lap 2's logged length
        Assert.Equal(new XRange(0, 100), stitched.LapRange);
        Assert.Equal(108, stitched.Series("Speed")[^1]);
    }

    [Fact]
    public void Stitch_uses_lap_times_as_the_offsets_by_time()
    {
        var stitched = StripChartBuilder.Stitch(LapWithNeighbours(), XAxisMode.Time);

        Assert.Equal(-5, stitched.X[0]);
        Assert.Equal(0, stitched.X[11]);
        Assert.Equal(4 + 2, stitched.X[^1]);
        Assert.Equal(new XRange(0, 4), stitched.LapRange);
    }

    [Fact]
    public void Markers_label_each_lap_start_and_the_initial_view_is_the_lap()
    {
        var fixture = TwoLapFixture();
        fixture.Laps[1] = LapWithNeighbours();

        var spec = StripChartBuilder.Build(fixture, XAxisMode.Distance);

        Assert.Equal(new XRange(0, 100), spec.InitialView);
        Assert.NotNull(spec.Markers);
        var labels = spec.Markers!.Select(m => (m.X, m.Label, m.Row)).ToList();
        // Lap 1 has no neighbours: a start and an end marker on row 0.
        Assert.Contains((0.0, "Lap 1", 0), labels);
        Assert.Contains((100.0, "Lap 1 end", 0), labels);
        // Lap 2: its predecessor, itself and its successor on row 1.
        Assert.Contains((-100.0, "Lap 1", 1), labels);
        Assert.Contains((0.0, "Lap 2", 1), labels);
        Assert.Contains((100.0, "Lap 3", 1), labels);
        Assert.Equal(5, labels.Count);
        Assert.Equal(new XRange(0, 100), spec.Subplots[0].Traces[1].Emphasis);
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
        var panel = StripChartBuilder.BuildDeltaTime(TwoLapFixture());
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
    public void Time_mode_has_no_delta_time_panel()
    {
        var spec = StripChartBuilder.Build(TwoLapFixture(), XAxisMode.Time);

        Assert.Single(spec.Subplots);
        Assert.Equal("Speed", spec.Subplots[0].Title);
        Assert.Equal("Time (s)", spec.XLabel);
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
}
