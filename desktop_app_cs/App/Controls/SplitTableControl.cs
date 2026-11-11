using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// A timing-screen style split table drawn directly: one row per lap, a
/// column per sector plus the lap time, the best time in each column in
/// purple and the second best in green, a theoretical-best row at the
/// bottom. Clicking a row raises <see cref="LapClicked"/>.
/// </summary>
public sealed class SplitTableControl : Control
{
    /// <summary>The table to draw.</summary>
    public static readonly StyledProperty<SplitTable?> TableProperty =
        AvaloniaProperty.Register<SplitTableControl, SplitTable?>(nameof(Table));

    /// <summary>Row to highlight.</summary>
    public static readonly StyledProperty<int?> HighlightedLapProperty =
        AvaloniaProperty.Register<SplitTableControl, int?>(nameof(HighlightedLap));

    private const double RowHeight = 24;
    private const double HeaderHeight = 28;
    private const double LapColumnWidth = 70;
    private const double SectorColumnWidth = 90;
    private const double LapTimeColumnWidth = 96;
    private static readonly Color Purple = Color.Parse("#a64dff");
    private static readonly Color Green = Color.Parse("#2ecc71");

    private readonly Typeface _typeface = new("Inter, Segoe UI, sans-serif");
    private readonly Typeface _mono = new("Consolas, Menlo, DejaVu Sans Mono, monospace");

    static SplitTableControl()
    {
        AffectsRender<SplitTableControl>(TableProperty, HighlightedLapProperty);
        AffectsMeasure<SplitTableControl>(TableProperty);
    }

    /// <summary>The table to draw.</summary>
    public SplitTable? Table
    {
        get => GetValue(TableProperty);
        set => SetValue(TableProperty, value);
    }

    /// <summary>Row to highlight.</summary>
    public int? HighlightedLap
    {
        get => GetValue(HighlightedLapProperty);
        set => SetValue(HighlightedLapProperty, value);
    }

    /// <summary>Raised with the lap number of a clicked row.</summary>
    public event Action<int>? LapClicked;

    /// <inheritdoc />
    protected override Size MeasureOverride(Size availableSize)
    {
        var table = Table;
        if (table is null)
        {
            return new Size(LapColumnWidth + LapTimeColumnWidth, HeaderHeight + RowHeight);
        }

        return new Size(LapColumnWidth + table.Track.SectorCount * SectorColumnWidth + LapTimeColumnWidth, HeaderHeight + (table.Laps.Count + 1) * RowHeight);
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        var table = Table;
        if (table is null)
        {
            return;
        }

        var row = (int)Math.Floor((e.GetPosition(this).Y - HeaderHeight) / RowHeight);
        if (row >= 0 && row < table.Laps.Count)
        {
            LapClicked?.Invoke(table.Laps[row].LapNum);
        }
    }

    private static string Time(double s) => double.IsNaN(s) ? "—" : s >= 60 ? $"{(int)(s / 60)}:{s - (int)(s / 60) * 60:00.000}" : s.ToString("0.000", CultureInfo.InvariantCulture);

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        var table = Table;
        var foreground = Foreground();
        var dim = new SolidColorBrush(foreground.Color, 0.55);
        if (table is null)
        {
            DrawText(context, "Select laps and a track to see split times", _typeface, 12, dim, new Point(8, 8));
            return;
        }

        var sectors = table.Track.SectorCount;
        var columns = new List<(string Label, double Width)> { ("Lap", LapColumnWidth) };
        columns.AddRange(table.Track.SectorNames.Select(n => (n, SectorColumnWidth)));
        columns.Add(("Lap time", LapTimeColumnWidth));

        // Second-best per column for the green colouring.
        var secondBest = new double[sectors + 1];
        for (var s = 0; s <= sectors; s++)
        {
            var values = table.Laps.Where(l => l.Complete).Select(l => s < sectors ? l.Sectors[s] : l.LapTimeS).OrderBy(v => v).Distinct().ToList();
            secondBest[s] = values.Count > 1 ? values[1] : double.NaN;
        }

        var bestLapTime = table.Laps.Where(l => l.Complete && l.LapTimeS > 0).Select(l => l.LapTimeS).DefaultIfEmpty(double.NaN).Min();

        // Header.
        var x = 0.0;
        var headerBrush = new SolidColorBrush(foreground.Color, 0.08);
        context.FillRectangle(headerBrush, new Rect(0, 0, columns.Sum(c => c.Width), HeaderHeight));
        foreach (var (label, width) in columns)
        {
            DrawText(context, label, _typeface, 12, foreground, new Point(x + 8, 7), FontWeight.Bold);
            x += width;
        }

        var y = HeaderHeight;
        var highlight = new SolidColorBrush(Color.Parse("#3b82f6"), 0.25);
        var stripe = new SolidColorBrush(foreground.Color, 0.04);
        for (var r = 0; r < table.Laps.Count; r++)
        {
            var lap = table.Laps[r];
            if (lap.LapNum == HighlightedLap)
            {
                context.FillRectangle(highlight, new Rect(0, y, columns.Sum(c => c.Width), RowHeight));
            }
            else if (r % 2 == 1)
            {
                context.FillRectangle(stripe, new Rect(0, y, columns.Sum(c => c.Width), RowHeight));
            }

            x = 0;
            DrawText(context, $"Lap {lap.LapNum}", _typeface, 12, foreground, new Point(x + 8, y + 5));
            x += LapColumnWidth;
            for (var s = 0; s < sectors; s++)
            {
                var value = lap.Sectors[s];
                var color = ColorFor(value, table.BestSector[s], secondBest[s], lap.Complete, foreground);
                DrawText(context, Time(value), _mono, 12, color, new Point(x + 8, y + 5));
                x += SectorColumnWidth;
            }

            var lapColor = ColorFor(lap.LapTimeS, bestLapTime, secondBest[sectors], lap.Complete, foreground);
            DrawText(context, Time(lap.LapTimeS), _mono, 12, lapColor, new Point(x + 8, y + 5), FontWeight.Bold);
            y += RowHeight;
        }

        // Theoretical best row.
        context.DrawLine(new Pen(dim, 1), new Point(0, y), new Point(columns.Sum(c => c.Width), y));
        x = 0;
        DrawText(context, "Best", _typeface, 12, dim, new Point(x + 8, y + 5), FontWeight.Bold);
        x += LapColumnWidth;
        var purple = new SolidColorBrush(Purple);
        for (var s = 0; s < sectors; s++)
        {
            DrawText(context, Time(table.BestSector[s]), _mono, 12, purple, new Point(x + 8, y + 5));
            x += SectorColumnWidth;
        }

        DrawText(context, Time(table.TheoreticalBestS), _mono, 12, purple, new Point(x + 8, y + 5), FontWeight.Bold);
    }

    private static IBrush ColorFor(double value, double best, double second, bool complete, SolidColorBrush foreground)
    {
        if (!complete || double.IsNaN(value))
        {
            return new SolidColorBrush(foreground.Color, 0.5);
        }

        if (value == best)
        {
            return new SolidColorBrush(Purple);
        }

        if (value == second)
        {
            return new SolidColorBrush(Green);
        }

        return foreground;
    }

    private SolidColorBrush Foreground()
    {
        var brush = this.FindResource("SystemControlForegroundBaseHighBrush") as ISolidColorBrush;
        return new SolidColorBrush(brush?.Color ?? Colors.Gray);
    }

    private static void DrawText(DrawingContext context, string text, Typeface typeface, double size, IBrush brush, Point origin, FontWeight weight = FontWeight.Normal)
    {
        var formatted = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface(typeface.FontFamily, FontStyle.Normal, weight), size, brush);
        context.DrawText(formatted, origin);
    }
}
