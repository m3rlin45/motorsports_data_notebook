using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;
using InfernoAnalyzer.Charting;
using ScottPlot;
using Color = Avalonia.Media.Color;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// The part of a strip-chart panel that changes with the pointer, drawn by
/// Avalonia above the plot: the panel's title tag, the crosshair, a marker
/// on each trace and the value read-outs. Keeping these out of the plot
/// means moving the cursor repaints this overlay only, never the traces.
/// </summary>
/// <remarks>
/// ScottPlot's Avalonia control renders in device-independent pixels with a
/// fixed layout, so data coordinates map to this control's own coordinate
/// space from the axis limits and the padding alone.
/// </remarks>
internal sealed class StripCursorOverlay : Control
{
    private const double TitleSize = 11;
    private const double ReadoutSize = 11;
    private const double ReadoutLineHeight = 16;
    private const double ReadoutGap = 8;
    private const double TitleRowHeight = 20;
    private const double SwatchWidth = 10;
    private const double MarkerRadius = 3.5;
    private const double FlipFraction = 0.75;
    private static readonly FontFeatureCollection TabularFigures = new() { FontFeature.Parse("tnum") };

    private readonly Plot _plot;
    private PixelPadding _padding;
    private ChartTheme _theme = ChartTheme.Dark;
    private string _title = string.Empty;
    private double? _cursor;
    private bool _pinned;
    private string _xLabel = string.Empty;
    private IReadOnlyList<Readout> _readouts = Array.Empty<Readout>();

    public StripCursorOverlay(Plot plot)
    {
        _plot = plot;
        IsHitTestVisible = false;
        ClipToBounds = true;
    }

    /// <summary>One trace's value at the cursor.</summary>
    /// <param name="Y">The value, in the panel's units.</param>
    /// <param name="Color">The trace's colour (hex).</param>
    /// <param name="Dashed">Whether the trace is drawn dashed.</param>
    /// <param name="Text">The read-out text, e.g. "L14 154.1 km/h".</param>
    public readonly record struct Readout(double Y, string Color, bool Dashed, string Text);

    /// <summary>The read-outs currently shown (for tests).</summary>
    public IReadOnlyList<Readout> Readouts => _readouts;

    /// <summary>The cursor position label on the x-axis, or empty (for tests).</summary>
    public string XLabel => _xLabel;

    /// <summary>Set the static parts: the title tag, the theme and the plot's fixed padding.</summary>
    public void Configure(string title, ChartTheme theme, PixelPadding padding)
    {
        _title = title;
        _theme = theme;
        _padding = padding;
        InvalidateVisual();
    }

    /// <summary>Move the cursor (null hides it) and replace the read-outs.</summary>
    public void SetCursor(double? x, bool pinned, string xLabel, IReadOnlyList<Readout> readouts)
    {
        _cursor = x;
        _pinned = pinned;
        _xLabel = xLabel;
        _readouts = readouts;
        InvalidateVisual();
    }

    /// <summary>The data area in this control's coordinates.</summary>
    private Rect DataRect => new(
        _padding.Left,
        _padding.Top,
        Math.Max(0, Bounds.Width - _padding.Left - _padding.Right),
        Math.Max(0, Bounds.Height - _padding.Top - _padding.Bottom));

    public override void Render(DrawingContext context)
    {
        var data = DataRect;
        if (data.Width <= 0 || data.Height <= 0)
        {
            return;
        }

        var family = TextElement.GetFontFamily(this);
        var ink = Color.Parse(_theme.Text);

        if (_title.Length > 0)
        {
            var title = Format(_title, new Typeface(family, Avalonia.Media.FontStyle.Normal, Avalonia.Media.FontWeight.SemiBold), TitleSize, new SolidColorBrush(ink, _theme.HighContrast ? 1 : 0.75));
            var box = new Rect(data.Left + 4, data.Top + 2, title.Width + 6, title.Height + 2);
            context.FillRectangle(new SolidColorBrush(Color.Parse(_theme.DataBackground), 0.8), box, 3);
            context.DrawText(title, new Point(box.Left + 3, box.Top + 1));
        }

        if (_cursor is not { } cursor)
        {
            return;
        }

        var xRange = _plot.Axes.Bottom.Range;
        var yRange = _plot.Axes.Left.Range;
        if (xRange.Span <= 0 || yRange.Span <= 0 || cursor < xRange.Min || cursor > xRange.Max)
        {
            return;
        }

        var px = data.Left + (cursor - xRange.Min) / xRange.Span * data.Width;
        var crosshair = new Pen(new SolidColorBrush(Color.Parse(_theme.Crosshair), 0.85), 1, _pinned ? null : DashStyle.Dash);
        context.DrawLine(crosshair, new Point(Math.Round(px) + 0.5, data.Top), new Point(Math.Round(px) + 0.5, data.Bottom));

        var background = Color.Parse(_theme.DataBackground);
        var ring = new Pen(new SolidColorBrush(background), 1.5);
        using (context.PushClip(data))
        {
            foreach (var readout in _readouts)
            {
                var py = data.Top + (yRange.Max - readout.Y) / yRange.Span * data.Height;
                context.DrawEllipse(new SolidColorBrush(Color.Parse(readout.Color)), ring, new Point(px, py), MarkerRadius, MarkerRadius);
            }
        }

        DrawReadouts(context, data, px, family, ink, background);

        if (_xLabel.Length > 0)
        {
            var label = Format(_xLabel, new Typeface(family), ReadoutSize, new SolidColorBrush(ink));
            var width = label.Width + 8;
            var left = Math.Clamp(px - width / 2, data.Left, Math.Max(data.Left, data.Right - width));
            var box = new Rect(left, data.Bottom + 2, width, label.Height + 2);
            context.DrawRectangle(new SolidColorBrush(Color.Parse(_theme.PanelBackground)), new Pen(new SolidColorBrush(Color.Parse(_theme.PanelOutline)), 1), box, 3, 3);
            context.DrawText(label, new Point(box.Left + 4, box.Top + 1));
        }
    }

    /// <summary>
    /// The read-outs stacked beside the crosshair from under the title tag,
    /// wrapping into further columns when there are more than fit the panel
    /// height, on the left of the crosshair once it is in the right quarter
    /// of the view so they never run off the panel.
    /// </summary>
    private void DrawReadouts(DrawingContext context, Rect data, double px, FontFamily family, Color ink, Color background)
    {
        if (_readouts.Count == 0)
        {
            return;
        }

        var typeface = new Typeface(family);
        var brush = new SolidColorBrush(ink);
        var texts = _readouts.Select(r => Format(r.Text, typeface, ReadoutSize, brush)).ToList();
        var top = data.Top + TitleRowHeight + 2;
        var perColumn = Math.Max(1, (int)((data.Bottom - top) / ReadoutLineHeight));
        var columnWidth = texts.Max(t => t.Width) + SwatchWidth + 14;
        var flip = px > data.Left + FlipFraction * data.Width;
        var backdrop = new SolidColorBrush(background, _theme.HighContrast ? 1 : 0.8);

        for (var i = 0; i < texts.Count; i++)
        {
            var column = i / perColumn;
            var row = i % perColumn;
            var left = flip
                ? px - ReadoutGap - (column + 1) * columnWidth
                : px + ReadoutGap + column * columnWidth;
            var y = top + row * ReadoutLineHeight;
            var box = new Rect(left, y, columnWidth - 2, ReadoutLineHeight - 1);
            context.FillRectangle(backdrop, box, 3);

            // The colour key is a mark beside the text; the text itself stays in ink.
            var key = new Pen(new SolidColorBrush(Color.Parse(_readouts[i].Color)), 2.5, _readouts[i].Dashed ? new DashStyle(new[] { 1.5, 1.0 }, 0) : null);
            var midY = y + (ReadoutLineHeight - 1) / 2;
            context.DrawLine(key, new Point(left + 4, midY), new Point(left + 4 + SwatchWidth, midY));
            context.DrawText(texts[i], new Point(left + SwatchWidth + 8, y + (ReadoutLineHeight - 1 - texts[i].Height) / 2));
        }
    }

    private static FormattedText Format(string text, Typeface typeface, double size, IBrush brush)
    {
        var formatted = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, size, brush);
        formatted.SetFontFeatures(TabularFigures);
        return formatted;
    }
}
