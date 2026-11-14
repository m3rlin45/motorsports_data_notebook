using System.Globalization;
using Avalonia;
using Avalonia.Automation.Peers;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using InfernoAnalyzer.Core.Tracks;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// A timing-screen style split table drawn directly: one row per lap, a
/// column per sector plus the lap time, the best time in each column in
/// purple and the second best in green (both also set apart by weight, so
/// the ranking does not depend on colour), a theoretical-best row at the
/// bottom. Click a row, or use Up/Down, to highlight a lap.
/// </summary>
/// <remarks>
/// Text comes from the inherited font properties and the colours from styled
/// properties (set from the app's theme resources in the Sectors view), so
/// the table follows the theme like any templated control.
/// </remarks>
public sealed class SplitTableControl : Control
{
    /// <summary>The table to draw.</summary>
    public static readonly StyledProperty<SplitTable?> TableProperty =
        AvaloniaProperty.Register<SplitTableControl, SplitTable?>(nameof(Table));

    /// <summary>Row to highlight.</summary>
    public static readonly StyledProperty<int?> HighlightedLapProperty =
        AvaloniaProperty.Register<SplitTableControl, int?>(nameof(HighlightedLap), defaultBindingMode: BindingMode.TwoWay);

    /// <summary>Text colour (inherited).</summary>
    public static readonly StyledProperty<IBrush?> ForegroundProperty = TextElement.ForegroundProperty.AddOwner<SplitTableControl>();

    /// <summary>Font (inherited).</summary>
    public static readonly StyledProperty<FontFamily> FontFamilyProperty = TextElement.FontFamilyProperty.AddOwner<SplitTableControl>();

    /// <summary>Font size (inherited).</summary>
    public static readonly StyledProperty<double> FontSizeProperty = TextElement.FontSizeProperty.AddOwner<SplitTableControl>();

    /// <summary>Colour of the best time in a column.</summary>
    public static readonly StyledProperty<IBrush?> BestBrushProperty =
        AvaloniaProperty.Register<SplitTableControl, IBrush?>(nameof(BestBrush), Brushes.MediumPurple);

    /// <summary>Colour of the second-best time in a column.</summary>
    public static readonly StyledProperty<IBrush?> SecondBestBrushProperty =
        AvaloniaProperty.Register<SplitTableControl, IBrush?>(nameof(SecondBestBrush), Brushes.SeaGreen);

    /// <summary>Colour of the highlighted row's bar and tint (the accent colour).</summary>
    public static readonly StyledProperty<IBrush?> HighlightBrushProperty =
        AvaloniaProperty.Register<SplitTableControl, IBrush?>(nameof(HighlightBrush), Brushes.DodgerBlue);

    /// <summary>Colour of the header and footer rules.</summary>
    public static readonly StyledProperty<IBrush?> RuleBrushProperty =
        AvaloniaProperty.Register<SplitTableControl, IBrush?>(nameof(RuleBrush), Brushes.Gray);

    private const double RowHeight = 24;
    private const double HeaderHeight = 28;
    private const double LapColumnWidth = 64;
    private const double SectorColumnWidth = 84;
    private const double LapTimeColumnWidth = 92;
    private const double CellPadding = 8;
    private static readonly FontFeatureCollection TabularFigures = new() { FontFeature.Parse("tnum") };

    static SplitTableControl()
    {
        AffectsRender<SplitTableControl>(
            TableProperty,
            HighlightedLapProperty,
            ForegroundProperty,
            FontFamilyProperty,
            FontSizeProperty,
            BestBrushProperty,
            SecondBestBrushProperty,
            HighlightBrushProperty,
            RuleBrushProperty);
        AffectsMeasure<SplitTableControl>(TableProperty);
        FocusableProperty.OverrideDefaultValue<SplitTableControl>(true);
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

    /// <summary>Colour of the best time in a column.</summary>
    public IBrush? BestBrush
    {
        get => GetValue(BestBrushProperty);
        set => SetValue(BestBrushProperty, value);
    }

    /// <summary>Colour of the second-best time in a column.</summary>
    public IBrush? SecondBestBrush
    {
        get => GetValue(SecondBestBrushProperty);
        set => SetValue(SecondBestBrushProperty, value);
    }

    /// <summary>Colour of the highlighted row's bar and tint.</summary>
    public IBrush? HighlightBrush
    {
        get => GetValue(HighlightBrushProperty);
        set => SetValue(HighlightBrushProperty, value);
    }

    /// <summary>Colour of the header and footer rules.</summary>
    public IBrush? RuleBrush
    {
        get => GetValue(RuleBrushProperty);
        set => SetValue(RuleBrushProperty, value);
    }

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

    private int RowAt(Point p)
    {
        var table = Table;
        if (table is null)
        {
            return -1;
        }

        var row = (int)Math.Floor((p.Y - HeaderHeight) / RowHeight);
        return row >= 0 && row < table.Laps.Count ? row : -1;
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        Focus();
        var row = RowAt(e.GetPosition(this));
        if (row >= 0 && Table is { } table)
        {
            var lap = table.Laps[row].LapNum;
            HighlightedLap = HighlightedLap == lap ? null : lap;
        }
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        Cursor = RowAt(e.GetPosition(this)) >= 0 ? new Cursor(StandardCursorType.Hand) : Cursor.Default;
    }

    /// <inheritdoc />
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (Table is not { Laps.Count: > 0 } table)
        {
            return;
        }

        var current = table.Laps.Select((l, i) => (l, i)).Where(x => x.l.LapNum == HighlightedLap).Select(x => x.i).DefaultIfEmpty(-1).First();
        switch (e.Key)
        {
            case Key.Down:
                HighlightedLap = table.Laps[Math.Min(current + 1, table.Laps.Count - 1)].LapNum;
                break;
            case Key.Up:
                HighlightedLap = table.Laps[Math.Max(current - 1, 0)].LapNum;
                break;
            case Key.Escape when HighlightedLap is not null:
                HighlightedLap = null;
                break;
            default:
                return;
        }

        e.Handled = true;
    }

    /// <inheritdoc />
    protected override AutomationPeer OnCreateAutomationPeer() =>
        new CustomDrawnPeer(this, AutomationControlType.Table, () => Table is { } t
            ? $"Split times: {t.Laps.Count} laps, {t.Track.SectorCount} sectors, theoretical best {Time(t.TheoreticalBestS)}"
            : "Split times: no data");

    private static string Time(double s) => double.IsNaN(s) ? "—" : s >= 60 ? $"{(int)(s / 60)}:{s - (int)(s / 60) * 60:00.000}" : s.ToString("0.000", CultureInfo.InvariantCulture);

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        var table = Table;
        var foreground = GetValue(ForegroundProperty) ?? Brushes.Gray;
        var dim = WithOpacity(foreground, 0.6);
        if (table is null)
        {
            Draw(context, "Select laps and a track to see split times", dim, new Rect(0, 0, Bounds.Width, RowHeight), right: false);
            return;
        }

        var sectors = table.Track.SectorCount;
        var columns = new List<(string Label, double Width)> { ("Lap", LapColumnWidth) };
        columns.AddRange(table.Track.SectorNames.Select(n => (n, SectorColumnWidth)));
        columns.Add(("Lap time", LapTimeColumnWidth));
        var width = columns.Sum(c => c.Width);

        // Second-best per column for the green colouring.
        var secondBest = new double[sectors + 1];
        for (var s = 0; s <= sectors; s++)
        {
            var values = table.Laps.Where(l => l.Complete).Select(l => s < sectors ? l.Sectors[s] : l.LapTimeS).OrderBy(v => v).Distinct().ToList();
            secondBest[s] = values.Count > 1 ? values[1] : double.NaN;
        }

        var bestLapTime = table.Laps.Where(l => l.Complete && l.LapTimeS > 0).Select(l => l.LapTimeS).DefaultIfEmpty(double.NaN).Min();
        var rule = new Pen(RuleBrush, 1);

        // Header: semibold labels over a rule; numeric columns right-aligned like their cells.
        var x = 0.0;
        for (var c = 0; c < columns.Count; c++)
        {
            Draw(context, columns[c].Label, foreground, new Rect(x, 0, columns[c].Width, HeaderHeight), right: c > 0, FontWeight.SemiBold);
            x += columns[c].Width;
        }

        context.DrawLine(rule, new Point(0, HeaderHeight - 0.5), new Point(width, HeaderHeight - 0.5));

        var y = HeaderHeight;
        foreach (var lap in table.Laps)
        {
            if (lap.LapNum == HighlightedLap && HighlightBrush is { } highlight)
            {
                context.FillRectangle(WithOpacity(highlight, 0.18), new Rect(0, y, width, RowHeight));
                context.FillRectangle(highlight, new Rect(0, y + 3, 3, RowHeight - 6), 1.5f);
            }

            x = 0;
            Draw(context, $"Lap {lap.LapNum}", foreground, new Rect(x, y, LapColumnWidth, RowHeight), right: false);
            x += LapColumnWidth;
            for (var s = 0; s < sectors; s++)
            {
                var (brush, weight) = Rank(lap.Sectors[s], table.BestSector[s], secondBest[s], lap.Complete, foreground, dim);
                Draw(context, Time(lap.Sectors[s]), brush, new Rect(x, y, SectorColumnWidth, RowHeight), right: true, weight);
                x += SectorColumnWidth;
            }

            var (lapBrush, lapWeight) = Rank(lap.LapTimeS, bestLapTime, secondBest[sectors], lap.Complete, foreground, dim);
            Draw(context, Time(lap.LapTimeS), lapBrush, new Rect(x, y, LapTimeColumnWidth, RowHeight), right: true, lapWeight);
            y += RowHeight;
        }

        // Theoretical best row.
        context.DrawLine(rule, new Point(0, y + 0.5), new Point(width, y + 0.5));
        x = 0;
        Draw(context, "Best", dim, new Rect(x, y, LapColumnWidth, RowHeight), right: false, FontWeight.SemiBold);
        x += LapColumnWidth;
        var best = BestBrush ?? foreground;
        for (var s = 0; s < sectors; s++)
        {
            Draw(context, Time(table.BestSector[s]), best, new Rect(x, y, SectorColumnWidth, RowHeight), right: true, FontWeight.SemiBold);
            x += SectorColumnWidth;
        }

        Draw(context, Time(table.TheoreticalBestS), best, new Rect(x, y, LapTimeColumnWidth, RowHeight), right: true, FontWeight.SemiBold);
    }

    /// <summary>Colour and weight of a time: best is bold purple, second best semibold green, incomplete laps dim.</summary>
    private (IBrush Brush, FontWeight Weight) Rank(double value, double best, double second, bool complete, IBrush foreground, IBrush dim)
    {
        if (!complete || double.IsNaN(value))
        {
            return (dim, FontWeight.Normal);
        }

        if (value == best)
        {
            return (BestBrush ?? foreground, FontWeight.Bold);
        }

        if (value == second)
        {
            return (SecondBestBrush ?? foreground, FontWeight.SemiBold);
        }

        return (foreground, FontWeight.Normal);
    }

    private static IBrush WithOpacity(IBrush brush, double opacity) =>
        brush is ISolidColorBrush solid ? new SolidColorBrush(solid.Color, solid.Opacity * opacity) : brush;

    private void Draw(DrawingContext context, string text, IBrush brush, Rect cell, bool right, FontWeight weight = FontWeight.Normal)
    {
        var typeface = new Typeface(GetValue(FontFamilyProperty), FontStyle.Normal, weight);
        var formatted = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, GetValue(FontSizeProperty), brush);
        formatted.SetFontFeatures(TabularFigures);
        var x = right ? cell.Right - CellPadding - formatted.Width : cell.Left + CellPadding;
        context.DrawText(formatted, new Point(x, cell.Top + (cell.Height - formatted.Height) / 2));
    }
}
