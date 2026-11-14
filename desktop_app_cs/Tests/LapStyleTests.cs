using InfernoAnalyzer.App;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

/// <summary>Lap styles: who gets which colour, and that nobody loses theirs to a newcomer.</summary>
public class LapStyleTests
{
    [Fact]
    public void Styles_run_through_the_hues_solid_then_dashed()
    {
        Assert.Equal(ChartColors.Lap(0), new LapStyle(0).Color);
        Assert.False(new LapStyle(7).Dashed);
        Assert.Equal(ChartColors.Lap(0), new LapStyle(8).Color);
        Assert.True(new LapStyle(8).Dashed);
        Assert.Equal(16, LapStyleAllocator.SlotCount);
    }

    [Fact]
    public void A_lap_keeps_its_style_while_selected_and_frees_it_when_dropped()
    {
        var styles = new LapStyleAllocator();
        styles.Assign("A", new[] { 14, 12 }, Array.Empty<int>());
        Assert.Equal(new LapStyle(0), styles.Get("A", 14));
        Assert.Equal(new LapStyle(1), styles.Get("A", 12));

        // Another lap joins: the two on screen do not move.
        styles.Assign("A", new[] { 14, 12, 10 }, Array.Empty<int>());
        Assert.Equal(new LapStyle(0), styles.Get("A", 14));
        Assert.Equal(new LapStyle(1), styles.Get("A", 12));
        Assert.Equal(new LapStyle(2), styles.Get("A", 10));

        // Dropping the best lap frees its slot without repainting the others; the next newcomer takes it.
        styles.Assign("A", new[] { 12, 10 }, Array.Empty<int>());
        Assert.Null(styles.Get("A", 14));
        Assert.Equal(new LapStyle(1), styles.Get("A", 12));
        styles.Assign("A", new[] { 12, 10, 9 }, Array.Empty<int>());
        Assert.Equal(new LapStyle(0), styles.Get("A", 9));
    }

    [Fact]
    public void Overlaid_laps_of_both_sessions_take_the_solid_hues_ahead_of_aggregated_laps()
    {
        var styles = new LapStyleAllocator();
        var changes = 0;
        styles.Changed += () => changes++;

        // Session A: two overlaid laps and ten more selected for the statistical tabs.
        styles.Assign("A", new[] { 14, 12 }, Enumerable.Range(1, 10).ToArray());
        Assert.Equal(12, Enumerable.Range(1, 14).Select(l => styles.Get("A", l)).Where(s => s is not null).Distinct().Count());

        // Session B loads: its overlaid laps get the next solid hues, not what thirteen A laps left over.
        styles.Assign("B", new[] { 7, 3 }, Array.Empty<int>());
        Assert.Equal(new LapStyle(2), styles.Get("B", 7));
        Assert.Equal(new LapStyle(3), styles.Get("B", 3));
        Assert.Equal(new LapStyle(0), styles.Get("A", 14));
        // The aggregated laps they displaced were moved, not dropped, and nothing collides.
        var all = Enumerable.Range(1, 14).Select(l => styles.Get("A", l)).Concat(new[] { styles.Get("B", 7), styles.Get("B", 3) }).Where(s => s is not null).ToList();
        Assert.Equal(14, all.Count);
        Assert.Equal(14, all.Distinct().Count());
        Assert.Equal(2, changes);
    }

    [Fact]
    public void More_laps_than_styles_reuse_them()
    {
        var styles = new LapStyleAllocator();
        styles.Assign("A", Array.Empty<int>(), Enumerable.Range(1, 20).ToArray());
        Assert.All(Enumerable.Range(1, 20), lap => Assert.NotNull(styles.Get("A", lap)));
    }
}
