using System.Collections.Specialized;
using System.ComponentModel;
using System.Globalization;
using Avalonia;
using Avalonia.Automation.Peers;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Input;
using Avalonia.Media;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// The track editor: the reference lap drawn to fit, gates as chords with
/// their names. Click the line to add a gate there, click a gate to select
/// it, drag a gate to move it along the line. With the keyboard: Left/Right
/// nudge the selected gate one sample (ten with Shift), Delete removes it,
/// Esc deselects.
/// </summary>
/// <remarks>
/// Colours are styled properties and text uses the inherited font, so the
/// editor follows the theme; it repaints only for the view model changes
/// that affect the drawing.
/// </remarks>
public sealed class TrackEditorControl : Control
{
    /// <summary>The Sectors view model whose track is edited.</summary>
    public static readonly StyledProperty<SectorsViewModel?> ViewModelProperty =
        AvaloniaProperty.Register<TrackEditorControl, SectorsViewModel?>(nameof(ViewModel));

    /// <summary>Text colour (inherited).</summary>
    public static readonly StyledProperty<IBrush?> ForegroundProperty = TextElement.ForegroundProperty.AddOwner<TrackEditorControl>();

    /// <summary>Font (inherited).</summary>
    public static readonly StyledProperty<FontFamily> FontFamilyProperty = TextElement.FontFamilyProperty.AddOwner<TrackEditorControl>();

    /// <summary>Colour of the track outline.</summary>
    public static readonly StyledProperty<IBrush?> TrackBrushProperty =
        AvaloniaProperty.Register<TrackEditorControl, IBrush?>(nameof(TrackBrush), Brushes.Gray);

    /// <summary>Colour of the gates.</summary>
    public static readonly StyledProperty<IBrush?> GateBrushProperty =
        AvaloniaProperty.Register<TrackEditorControl, IBrush?>(nameof(GateBrush), Brushes.DarkOrange);

    /// <summary>Colour of the selected gate (the accent colour).</summary>
    public static readonly StyledProperty<IBrush?> SelectedGateBrushProperty =
        AvaloniaProperty.Register<TrackEditorControl, IBrush?>(nameof(SelectedGateBrush), Brushes.DodgerBlue);

    private const double EdgeMargin = 24;
    private const double HitRadius = 10;
    private const double DragSlop = 3;
    private static readonly Cursor HandCursor = new(StandardCursorType.Hand);
    private static readonly Cursor CrossCursor = new(StandardCursorType.Cross);

    private SectorsViewModel? _bound;
    private bool _dragging;
    private bool _moved;
    private Point _pressedAt;

    static TrackEditorControl()
    {
        AffectsRender<TrackEditorControl>(ViewModelProperty, ForegroundProperty, FontFamilyProperty, TrackBrushProperty, GateBrushProperty, SelectedGateBrushProperty);
        FocusableProperty.OverrideDefaultValue<TrackEditorControl>(true);
    }

    /// <summary>The Sectors view model whose track is edited.</summary>
    public SectorsViewModel? ViewModel
    {
        get => GetValue(ViewModelProperty);
        set => SetValue(ViewModelProperty, value);
    }

    /// <summary>Colour of the track outline.</summary>
    public IBrush? TrackBrush
    {
        get => GetValue(TrackBrushProperty);
        set => SetValue(TrackBrushProperty, value);
    }

    /// <summary>Colour of the gates.</summary>
    public IBrush? GateBrush
    {
        get => GetValue(GateBrushProperty);
        set => SetValue(GateBrushProperty, value);
    }

    /// <summary>Colour of the selected gate.</summary>
    public IBrush? SelectedGateBrush
    {
        get => GetValue(SelectedGateBrushProperty);
        set => SetValue(SelectedGateBrushProperty, value);
    }

    /// <inheritdoc />
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == ViewModelProperty)
        {
            Rebind();
        }
    }

    private void Rebind()
    {
        if (_bound is not null)
        {
            _bound.PropertyChanged -= OnVmChanged;
            _bound.Gates.CollectionChanged -= OnGatesChanged;
            WatchGates(_bound.Gates, watch: false);
        }

        _bound = ViewModel;
        if (_bound is not null)
        {
            _bound.PropertyChanged += OnVmChanged;
            _bound.Gates.CollectionChanged += OnGatesChanged;
            WatchGates(_bound.Gates, watch: true);
        }
    }

    private void WatchGates(IEnumerable<GateViewModel> gates, bool watch)
    {
        foreach (var gate in gates)
        {
            gate.PropertyChanged -= OnGateChanged;
            if (watch)
            {
                gate.PropertyChanged += OnGateChanged;
            }
        }
    }

    private void OnVmChanged(object? sender, PropertyChangedEventArgs e)
    {
        // Status, the table and the dirty flag change constantly and draw nothing here.
        if (e.PropertyName is nameof(SectorsViewModel.ReferencePath) or nameof(SectorsViewModel.SelectedGate) or nameof(SectorsViewModel.SelectedTrack))
        {
            InvalidateVisual();
        }
    }

    private void OnGatesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        if (_bound is not null)
        {
            WatchGates(_bound.Gates, watch: true);
        }

        InvalidateVisual();
    }

    private void OnGateChanged(object? sender, PropertyChangedEventArgs e) => InvalidateVisual();

    /// <summary>Scale/offset mapping local metres to the control.</summary>
    private (double Scale, double OffsetX, double OffsetY)? Transform()
    {
        if (ViewModel?.ReferencePath is not { } path || path.X.Length == 0 || Bounds.Width <= 2 * EdgeMargin || Bounds.Height <= 2 * EdgeMargin)
        {
            return null;
        }

        var minX = path.X.Min();
        var maxX = path.X.Max();
        var minY = path.Y.Min();
        var maxY = path.Y.Max();
        var scale = Math.Min((Bounds.Width - 2 * EdgeMargin) / Math.Max(maxX - minX, 1), (Bounds.Height - 2 * EdgeMargin) / Math.Max(maxY - minY, 1));
        var offsetX = EdgeMargin + ((Bounds.Width - 2 * EdgeMargin) - (maxX - minX) * scale) / 2 - minX * scale;
        var offsetY = EdgeMargin + ((Bounds.Height - 2 * EdgeMargin) - (maxY - minY) * scale) / 2 + maxY * scale;
        return (scale, offsetX, offsetY);
    }

    private static Point ToScreen((double Scale, double OffsetX, double OffsetY) t, double x, double y) =>
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

    private bool IsOnLine(Point p) =>
        ViewModel is { HasTrack: true } && NearestSample(p) is { } near && near.DistancePx <= HitRadius * 2;

    /// <summary>Apply a press at <paramref name="p"/>: select a gate, or add one on the line.</summary>
    public void Press(Point p)
    {
        if (ViewModel is null)
        {
            return;
        }

        _pressedAt = p;
        _moved = false;
        if (GateAt(p) is { } gate)
        {
            ViewModel.SelectedGate = gate;
            _dragging = true;
            return;
        }

        if (IsOnLine(p) && NearestSample(p) is { } near)
        {
            ViewModel.AddGateAt(near.Index);
            _dragging = true;
            _moved = true; // adding was already an undo step; dragging the new gate into place belongs to it
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
        Focus();
        Press(e.GetPosition(this));
        e.Handled = true;
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var p = e.GetPosition(this);
        if (!_dragging)
        {
            Cursor = GateAt(p) is not null ? HandCursor : IsOnLine(p) ? CrossCursor : Cursor.Default;
            return;
        }

        if (ViewModel?.SelectedGate is null || NearestSample(p) is not { } near)
        {
            return;
        }

        if (!_moved)
        {
            if (Math.Abs(p.X - _pressedAt.X) < DragSlop && Math.Abs(p.Y - _pressedAt.Y) < DragSlop)
            {
                return; // a click that wobbled, not a drag
            }

            _moved = true;
            ViewModel.BeginGateMove();
        }

        ViewModel.MoveSelectedGateTo(near.Index);
    }

    /// <inheritdoc />
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _dragging = false;
    }

    /// <inheritdoc />
    protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e)
    {
        base.OnPointerCaptureLost(e);
        _dragging = false; // the release went elsewhere (another window took the pointer); do not keep dragging on hover
    }

    /// <inheritdoc />
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (ViewModel is not { SelectedGate: not null } vm)
        {
            return;
        }

        var step = e.KeyModifiers.HasFlag(KeyModifiers.Shift) ? 10 : 1;
        switch (e.Key)
        {
            case Key.Left:
                vm.NudgeSelectedGate(-step);
                break;
            case Key.Right:
                vm.NudgeSelectedGate(step);
                break;
            case Key.Delete or Key.Back:
                vm.DeleteGate();
                break;
            case Key.Escape:
                vm.SelectedGate = null;
                break;
            default:
                return;
        }

        e.Handled = true;
    }

    /// <inheritdoc />
    protected override AutomationPeer OnCreateAutomationPeer() =>
        new CustomDrawnPeer(this, AutomationControlType.Group, () => ViewModel is { HasTrack: true } vm
            ? $"Track editor: {vm.TrackName}, {vm.Gates.Count} gates" + (vm.SelectedGate is { } g ? $", {g.Name} selected" : string.Empty)
            : "Track editor: no track");

    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        // Fill the bounds so the whole area takes pointer input, not only the drawn pixels.
        context.FillRectangle(Brushes.Transparent, new Rect(Bounds.Size));
        var foreground = GetValue(ForegroundProperty) ?? Brushes.Gray;
        var family = GetValue(FontFamilyProperty);
        if (ViewModel?.ReferencePath is not { } path || path.X.Length == 0 || Transform() is not { } t)
        {
            return; // the view shows the empty-state text over this control
        }

        var line = new Pen(TrackBrush, 2, lineCap: PenLineCap.Round, lineJoin: PenLineJoin.Round);
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

        // Start/finish: a ringed dot in the text colour with its label.
        var start = ToScreen(t, path.X[0], path.Y[0]);
        context.DrawEllipse(foreground, null, start, 4.5, 4.5);
        var startLabel = new FormattedText("S/F", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface(family), 11, foreground);
        context.DrawText(startLabel, new Point(start.X + 8, start.Y - startLabel.Height / 2));

        foreach (var gate in ViewModel.Gates)
        {
            if (ViewModel.GateLocal(gate) is not { } local)
            {
                continue;
            }

            var a = ToScreen(t, local.A.X, local.A.Y);
            var b = ToScreen(t, local.B.X, local.B.Y);
            var selected = ReferenceEquals(gate, ViewModel.SelectedGate);
            var brush = (selected ? SelectedGateBrush : GateBrush) ?? foreground;
            context.DrawLine(new Pen(brush, selected ? 4 : 3, lineCap: PenLineCap.Round), a, b);
            // The label is ink; the chord beside it carries the colour.
            var label = new FormattedText(gate.Name, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface(family, FontStyle.Normal, selected ? FontWeight.Bold : FontWeight.SemiBold), 11, foreground);
            context.DrawText(label, new Point(Math.Max(a.X, b.X) + 5, Math.Min(a.Y, b.Y) - label.Height / 2 - 4));
        }
    }
}
