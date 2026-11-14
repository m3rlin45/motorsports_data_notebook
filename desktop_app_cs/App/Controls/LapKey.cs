using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// The legend key of a lap: a short stroke in the colour and dash pattern
/// its trace is drawn with, resolved for the current theme. Draws nothing
/// while the lap has no style, keeping the row's layout.
/// </summary>
public sealed class LapKey : Control
{
    /// <summary>The lap's trace style; null draws nothing.</summary>
    public static readonly StyledProperty<LapStyle?> LapStyleProperty =
        AvaloniaProperty.Register<LapKey, LapStyle?>(nameof(LapStyle));

    static LapKey()
    {
        AffectsRender<LapKey>(LapStyleProperty);
    }

    /// <summary>Create the key.</summary>
    public LapKey()
    {
        Width = 18;
        Height = 12;
        ActualThemeVariantChanged += (_, _) => InvalidateVisual();
    }

    /// <summary>The lap's trace style; null draws nothing.</summary>
    public LapStyle? LapStyle
    {
        get => GetValue(LapStyleProperty);
        set => SetValue(LapStyleProperty, value);
    }

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        if (LapStyle is not { } style)
        {
            return;
        }

        var color = Color.Parse(ChartSurfaces.ThemeFor(this).Resolve(style.Color));
        var pen = new Pen(new SolidColorBrush(color), 3, style.Dashed ? new DashStyle(new[] { 1.5, 1.0 }, 0) : null);
        var y = Bounds.Height / 2;
        context.DrawLine(pen, new Point(1, y), new Point(Bounds.Width - 1, y));
    }
}
