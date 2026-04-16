using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.LogicalTree;
using Avalonia.Media.Transformation;
using Avalonia.Threading;

namespace TirePressureCalculator.Views;

public partial class MainView : UserControl
{
    // Aspect ratio thresholds for selecting between the three CornerTemplate
    // layouts (narrow-stacked, medium-triangle, wide-row).
    private const double NarrowThreshold = 1.0;
    private const double WideThreshold = 1.7;

    // Android standard tablet threshold: if the smallest dimension is >= 600 DIPs,
    // we treat the device as tablet-sized.
    private const double PhoneSizeThreshold = 600.0;

    // How much to scale the focused quadrant on phone when an input is tapped.
    private const double PhoneZoomScale = 2.0;

    public static readonly StyledProperty<bool> IsNarrowProperty =
        AvaloniaProperty.Register<MainView, bool>(nameof(IsNarrow));

    public static readonly StyledProperty<bool> IsWideProperty =
        AvaloniaProperty.Register<MainView, bool>(nameof(IsWide));

    public bool IsNarrow
    {
        get => GetValue(IsNarrowProperty);
        private set => SetValue(IsNarrowProperty, value);
    }

    public bool IsWide
    {
        get => GetValue(IsWideProperty);
        private set => SetValue(IsWideProperty, value);
    }

    public bool IsMedium => !IsNarrow && !IsWide;

    public static readonly DirectProperty<MainView, bool> IsMediumProperty =
        AvaloniaProperty.RegisterDirect<MainView, bool>(nameof(IsMedium), o => o.IsMedium);

    // True when the smallest screen dimension is below the tablet threshold.
    // Drives the phone-only zoom-to-quadrant behavior.
    public bool IsPhoneSize { get; private set; } = true;

    private ContentControl? _zoomedQuadrant;

    public MainView()
    {
        InitializeComponent();
        AddHandler(GotFocusEvent, OnChildGotFocus);
        AddHandler(LostFocusEvent, OnChildLostFocus);
        AddHandler(KeyDownEvent, OnChildKeyDown, RoutingStrategies.Bubble, handledEventsToo: true);
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == BoundsProperty)
        {
            var b = Bounds;
            if (b.Height <= 0) return;

            double aspect = b.Width / b.Height;
            bool wasMedium = IsMedium;
            IsNarrow = aspect < NarrowThreshold;
            IsWide = aspect > WideThreshold;
            if (IsMedium != wasMedium)
            {
                RaisePropertyChanged(IsMediumProperty, wasMedium, IsMedium);
            }

            IsPhoneSize = System.Math.Min(b.Width, b.Height) < PhoneSizeThreshold;
        }
    }

    private void OnChildGotFocus(object? sender, GotFocusEventArgs e)
    {
        // Only adjust layout on single-view lifetimes (Android). Desktop ignores this.
        if (Application.Current?.ApplicationLifetime is not ISingleViewApplicationLifetime) return;

        var quadrant = FindQuadrant(e.Source as Control);
        if (quadrant is null) return;

        if (IsPhoneSize)
        {
            ZoomTo(quadrant);
        }
        else
        {
            // Tablet: scroll the rear row above the keyboard; front stays put.
            if (quadrant == RLCorner || quadrant == RRCorner)
            {
                ScrollRearIntoView();
            }
            else
            {
                RootScroll.Offset = default;
            }
        }
    }

    private void OnChildLostFocus(object? sender, RoutedEventArgs e)
    {
        if (Application.Current?.ApplicationLifetime is not ISingleViewApplicationLifetime) return;

        // Defer until the next focus event lands — a focus shift between two
        // inputs in the same quadrant should not trigger an unzoom/unscroll.
        Dispatcher.UIThread.Post(() =>
        {
            var focused = TopLevel.GetTopLevel(this)?.FocusManager?.GetFocusedElement() as Control;
            var quadrant = FindQuadrant(focused);
            if (quadrant is null)
            {
                // Focus left the quadrants entirely — reset.
                if (_zoomedQuadrant is not null) ResetZoom();
                RootScroll.Offset = default;
            }
            else if (IsPhoneSize && quadrant != _zoomedQuadrant)
            {
                ZoomTo(quadrant);
            }
            else if (!IsPhoneSize)
            {
                if (quadrant == RLCorner || quadrant == RRCorner)
                    ScrollRearIntoView();
                else
                    RootScroll.Offset = default;
            }
        });
    }

    private void OnChildKeyDown(object? sender, KeyEventArgs e)
    {
        // Enter on the soft keyboard should dismiss focus, which chains into
        // OnChildLostFocus and reverses the zoom/scroll.
        if (e.Key == Key.Enter || e.Key == Key.Return)
        {
            Focus();
        }
    }

    private ContentControl? FindQuadrant(Control? control)
    {
        while (control is not null)
        {
            if (control == FLCorner || control == FRCorner ||
                control == RLCorner || control == RRCorner)
            {
                return (ContentControl)control;
            }
            control = (control as ILogical)?.LogicalParent as Control
                     ?? control.Parent as Control;
        }
        return null;
    }

    private void ZoomTo(ContentControl quadrant)
    {
        // Set the transform origin to the center of the target quadrant so a
        // uniform scale visually zooms into it. Computing origin from the
        // quadrant's actual bounds keeps this correct regardless of aspect
        // ratio (the 2x2 grid changes proportions as the window resizes).
        var gridBounds = RootGrid.Bounds;
        var origin = quadrant.TranslatePoint(
            new Point(quadrant.Bounds.Width / 2, quadrant.Bounds.Height / 2),
            RootGrid) ?? new Point(gridBounds.Width / 2, gridBounds.Height / 2);

        double originX = gridBounds.Width > 0 ? origin.X / gridBounds.Width : 0.5;
        double originY = gridBounds.Height > 0 ? origin.Y / gridBounds.Height : 0.5;

        RootGrid.RenderTransformOrigin = new RelativePoint(originX, originY, RelativeUnit.Relative);
        RootGrid.RenderTransform = TransformOperations.Parse($"scale({PhoneZoomScale})");
        _zoomedQuadrant = quadrant;
    }

    private void ResetZoom()
    {
        RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
        _zoomedQuadrant = null;
    }

    private void ScrollRearIntoView()
    {
        // Translate the rear row's origin into ScrollViewer coordinates. This
        // is how far to scroll so the rear row sits at the top of the visible
        // area, safely above the keyboard.
        var rlTop = RLCorner.TranslatePoint(new Point(0, 0), RootScroll);
        if (rlTop.HasValue && rlTop.Value.Y > 0)
        {
            RootScroll.Offset = new Vector(0, rlTop.Value.Y);
        }
    }
}
