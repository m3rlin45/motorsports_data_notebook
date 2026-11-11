using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// The track editor: the reference lap drawn to fit, gates as chords with
/// their names. Click the line to add a gate there, click a gate to select
/// it, drag a gate to move it along the line.
/// </summary>
public sealed class TrackEditorControl : Control
{
    /// <summary>The Sectors view model whose track is edited.</summary>
    public static readonly StyledProperty<SectorsViewModel?> ViewModelProperty =
        AvaloniaProperty.Register<TrackEditorControl, SectorsViewModel?>(nameof(ViewModel));

    private const double Margin = 24;
    private const double HitRadius = 10;
    private SectorsViewModel? _bound;
    private bool _dragging;

    static TrackEditorControl()
    {
        AffectsRender<TrackEditorControl>(ViewModelProperty);
    }

    /// <summary>Create the control.</summary>
    public TrackEditorControl()
    {
        PropertyChanged += (_, e) =>
        {
            if (e.Property == ViewModelProperty)
            {
                Rebind();
            }
        };
    }

    /// <summary>The Sectors view model whose track is edited.</summary>
    public SectorsViewModel? ViewModel
    {
        get => GetValue(ViewModelProperty);
        set => SetValue(ViewModelProperty, value);
    }

    private void Rebind()
    {
        if (_bound is not null)
        {
            _bound.PropertyChanged -= OnVmChanged;
            _bound.Gates.CollectionChanged -= OnGatesChanged;
        }

        _bound = ViewModel;
        if (_bound is not null)
        {
            _bound.PropertyChanged += OnVmChanged;
            _bound.Gates.CollectionChanged += OnGatesChanged;
        }

        InvalidateVisual();
    }

    private void OnVmChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e) => InvalidateVisual();

    private void OnGatesChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) => InvalidateVisual();

    /// <summary>Scale/offset mapping local metres to the control.</summary>
    private (double Scale, double OffsetX, double OffsetY)? Transform()
    {
        if (ViewModel?.ReferencePath is not { } path || path.X.Length == 0 || Bounds.Width <= 2 * Margin || Bounds.Height <= 2 * Margin)
        {
            return null;
        }

        var minX = path.X.Min();
        var maxX = path.X.Max();
        var minY = path.Y.Min();
        var maxY = path.Y.Max();
        var scale = Math.Min((Bounds.Width - 2 * Margin) / Math.Max(maxX - minX, 1), (Bounds.Height - 2 * Margin) / Math.Max(maxY - minY, 1));
        var offsetX = Margin + ((Bounds.Width - 2 * Margin) - (maxX - minX) * scale) / 2 - minX * scale;
        var offsetY = Margin + ((Bounds.Height - 2 * Margin) - (maxY - minY) * scale) / 2 + maxY * scale;
        return (scale, offsetX, offsetY);
    }

    private Point ToScreen((double Scale, double OffsetX, double OffsetY) t, double x, double y) =>
        new(t.OffsetX + x * t.Scale, t.OffsetY - y * t.Scale);

    /// <summary>Index of the path sample nearest to a screen point, and its distance in pixels.</summary>
    public (int Index, double DistancePx)? NearestSample(Point p)
    {
        if (ViewModel?.ReferencePath is not { } path || Transform() is not { } t)
        {
            return null;
        }

        var best = 0;
        var bestD2 = double.PositiveInfinity;
        for (var i = 0; i < path.X.Length; i++)
        {
            var s = ToScreen(t, path.X[i], path.Y[i]);
            var d2 = (s.X - p.X) * (s.X - p.X) + (s.Y - p.Y) * (s.Y - p.Y);
            if (d2 < bestD2)
            {
                bestD2 = d2;
                best = i;
            }
        }

        return (best, Math.Sqrt(bestD2));
    }

    private GateViewModel? GateAt(Point p)
    {
        if (ViewModel is null || Transform() is not { } t)
        {
            return null;
        }

        foreach (var gate in ViewModel.Gates)
        {
            if (ViewModel.GateLocal(gate) is not { } local)
            {
                continue;
            }

            var a = ToScreen(t, local.A.X, local.A.Y);
            var b = ToScreen(t, local.B.X, local.B.Y);
            if (DistanceToSegment(p, a, b) <= HitRadius)
            {
                return gate;
            }
        }

        return null;
    }

    private static double DistanceToSegment(Point p, Point a, Point b)
    {
        var abx = b.X - a.X;
        var aby = b.Y - a.Y;
        var len2 = abx * abx + aby * aby;
        var u = len2 > 0 ? Math.Clamp(((p.X - a.X) * abx + (p.Y - a.Y) * aby) / len2, 0, 1) : 0;
        var cx = a.X + u * abx;
        var cy = a.Y + u * aby;
        return Math.Sqrt((p.X - cx) * (p.X - cx) + (p.Y - cy) * (p.Y - cy));
    }

    /// <summary>Apply a press at <paramref name="p"/>: select a gate, or add one on the line.</summary>
    public void Press(Point p)
    {
        if (ViewModel is null)
        {
            return;
        }

        if (GateAt(p) is { } gate)
        {
            ViewModel.SelectedGate = gate;
            _dragging = true;
            return;
        }

        if (NearestSample(p) is { } near && near.DistancePx <= HitRadius * 2 && ViewModel.HasTrack)
        {
            ViewModel.AddGateAt(near.Index);
            _dragging = true;
        }
        else
        {
            ViewModel.SelectedGate = null;
        }
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        Press(e.GetPosition(this));
        e.Handled = true;
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (_dragging && ViewModel?.SelectedGate is not null && NearestSample(e.GetPosition(this)) is { } near)
        {
            ViewModel.MoveSelectedGateTo(near.Index);
        }
    }

    /// <inheritdoc />
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _dragging = false;
    }

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        var fg = (this.FindResource("SystemControlForegroundBaseHighBrush") as ISolidColorBrush)?.Color ?? Colors.Gray;
        var foreground = new SolidColorBrush(fg);
        if (ViewModel?.ReferencePath is not { } path || Transform() is not { } t)
        {
            var hint = new FormattedText("Load a session with GPS and select laps to edit the track", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Inter, Segoe UI, sans-serif"), 12, new SolidColorBrush(fg, 0.55));
            context.DrawText(hint, new Point(8, 8));
            return;
        }

        var line = new Pen(new SolidColorBrush(fg, 0.7), 2);
        var geometry = new StreamGeometry();
        using (var g = geometry.Open())
        {
            g.BeginFigure(ToScreen(t, path.X[0], path.Y[0]), false);
            for (var i = 1; i < path.X.Length; i++)
            {
                g.LineTo(ToScreen(t, path.X[i], path.Y[i]));
            }

            g.EndFigure(false);
        }

        context.DrawGeometry(null, line, geometry);

        // Start/finish marker.
        var start = ToScreen(t, path.X[0], path.Y[0]);
        context.DrawEllipse(new SolidColorBrush(Color.Parse("#2ecc71")), null, start, 5, 5);

        foreach (var gate in ViewModel.Gates)
        {
            if (ViewModel.GateLocal(gate) is not { } local)
            {
                continue;
            }

            var a = ToScreen(t, local.A.X, local.A.Y);
            var b = ToScreen(t, local.B.X, local.B.Y);
            var selected = ReferenceEquals(gate, ViewModel.SelectedGate);
            var color = selected ? Color.Parse("#f59e0b") : Color.Parse("#3b82f6");
            context.DrawLine(new Pen(new SolidColorBrush(color), selected ? 4 : 3), a, b);
            var label = new FormattedText(gate.Name, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Inter, Segoe UI, sans-serif", FontStyle.Normal, FontWeight.Bold), 11, new SolidColorBrush(color));
            context.DrawText(label, new Point(Math.Max(a.X, b.X) + 4, Math.Min(a.Y, b.Y) - 6));
        }
    }
}
