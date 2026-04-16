using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Platform;
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
    private Control? _lastFocusedInput;
    private bool _tabletTranslated;

    // Static reference so the Android Activity can ask us to handle Back.
    public static MainView? Instance { get; private set; }

    public MainView()
    {
        InitializeComponent();
        Instance = this;
        AddHandler(GotFocusEvent, OnChildGotFocus);
        AddHandler(LostFocusEvent, OnChildLostFocus);
        AddHandler(KeyDownEvent, OnChildKeyDown, RoutingStrategies.Bubble, handledEventsToo: true);
    }

    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);

        // Subscribe to InputPane (soft keyboard) state changes so we can
        // unzoom when the keyboard is dismissed (Done button or Back key).
        var inputPane = TopLevel.GetTopLevel(this)?.InputPane;
        if (inputPane is not null)
            inputPane.StateChanged += OnInputPaneStateChanged;
    }

    private void OnInputPaneStateChanged(object? sender, InputPaneStateEventArgs e)
    {
        if (Application.Current?.ApplicationLifetime is not ISingleViewApplicationLifetime) return;

        if (e.NewState == InputPaneState.Open)
        {
            // Tablet: translate the grid up so rear fields sit above the keyboard.
            if (!IsPhoneSize)
            {
                Dispatcher.UIThread.Post(() =>
                {
                    var focused = TopLevel.GetTopLevel(this)?.FocusManager?
                        .GetFocusedElement() as Control;
                    var quadrant = FindQuadrant(focused);
                    if (quadrant == RLCorner || quadrant == RRCorner)
                        TranslateForRear();
                });
            }
            return;
        }

        // Closed — keyboard dismissed (by Done, Back, or system).
        // Always reset: ReturnKeyType="Next" on fields 1-2 keeps the
        // keyboard open when advancing, so Closed only fires for the
        // last field's Done or a Back press — both should unzoom.
        if (_tabletTranslated)
        {
            RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
            _tabletTranslated = false;
        }
        if (_zoomedQuadrant is not null)
        {
            ResetZoom();
        }
        Focus();
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
        var source = e.Source as Control;

        // Select all text so new input replaces the existing value.
        // For touch, PointerReleased fires after GotFocus and repositions
        // the caret, so we delay past it. For keyboard (Tab), run immediately.
        if (source is TextBox tb)
        {
            if (e.NavigationMethod == NavigationMethod.Pointer)
                DispatcherTimer.RunOnce(() => tb.SelectAll(), System.TimeSpan.FromMilliseconds(50));
            else
                Dispatcher.UIThread.Post(() => tb.SelectAll());
        }

        // Only adjust layout on single-view lifetimes (Android). Desktop ignores this.
        if (Application.Current?.ApplicationLifetime is not ISingleViewApplicationLifetime) return;

        _lastFocusedInput = source;
        var quadrant = FindQuadrant(source);
        if (quadrant is null) return;

        if (IsPhoneSize && quadrant != _zoomedQuadrant)
        {
            ZoomTo(quadrant);
        }
        else if (!IsPhoneSize)
        {
            // Tablet: translate the grid up if rear fields are behind the keyboard.
            var inputPane = TopLevel.GetTopLevel(this)?.InputPane;
            if (quadrant == RLCorner || quadrant == RRCorner)
            {
                if (inputPane?.State == InputPaneState.Open)
                    TranslateForRear();
            }
            else if (_tabletTranslated)
            {
                RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
                _tabletTranslated = false;
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
                if (_tabletTranslated)
                {
                    RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
                    _tabletTranslated = false;
                }
            }
            else if (IsPhoneSize && quadrant != _zoomedQuadrant)
            {
                ZoomTo(quadrant);
            }
            else if (!IsPhoneSize)
            {
                var inputPane = TopLevel.GetTopLevel(this)?.InputPane;
                if (quadrant == RLCorner || quadrant == RRCorner)
                {
                    if (inputPane?.State == InputPaneState.Open)
                        TranslateForRear();
                }
                else if (_tabletTranslated)
                {
                    RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
                    _tabletTranslated = false;
                }
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

    private static T? FindAncestorOfType<T>(Control? control) where T : Control
    {
        while (control is not null)
        {
            if (control is T match) return match;
            control = (control as ILogical)?.LogicalParent as Control
                     ?? control.Parent as Control;
        }
        return null;
    }

    private void ZoomTo(ContentControl quadrant)
    {
        var gridBounds = RootGrid.Bounds;
        if (gridBounds.Width <= 0 || gridBounds.Height <= 0) return;

        // Compute the quadrant's top-left position in grid coordinates.
        var topLeft = quadrant.TranslatePoint(new Point(0, 0), RootGrid)
                      ?? new Point(0, 0);

        // X origin: pin the quadrant's outer edge so it fills the viewport width.
        //   Left quadrants (FL/RL): left edge stays at x=0  → originX = 0
        //   Right quadrants (FR/RR): right edge stays at x=1 → originX = 1
        bool isRight = quadrant == FRCorner || quadrant == RRCorner;
        double originX = isRight ? 1.0 : 0.0;

        // Y origin: pin the quadrant's top edge to the viewport top (y=0).
        // With scale s from origin o, point p maps to: o + (p − o) × s.
        // Setting the mapped quadrant-top to 0 and solving gives:
        //   originY = s × quadTopRel / (s − 1)
        double quadTopRel = topLeft.Y / gridBounds.Height;
        double originY = PhoneZoomScale * quadTopRel / (PhoneZoomScale - 1);

        RootGrid.RenderTransformOrigin = new RelativePoint(
            originX, originY, RelativeUnit.Relative);
        RootGrid.RenderTransform = TransformOperations.Parse($"scale({PhoneZoomScale})");
        _zoomedQuadrant = quadrant;
    }

    /// <summary>
    /// Called by the Android Activity when the Back button is pressed.
    /// Returns true if the back was consumed (zoom reset), false to
    /// let the system handle it (close the app).
    /// </summary>
    public bool TryHandleBack()
    {
        if (_zoomedQuadrant is not null)
        {
            ResetZoom();
            Focus();
            return true;
        }
        if (_tabletTranslated)
        {
            RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
            _tabletTranslated = false;
            Focus();
            return true;
        }
        return false;
    }

    private void ResetZoom()
    {
        RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
        _zoomedQuadrant = null;
    }

    private void TranslateForRear()
    {
        // Use InputPane.OccludedRect to find exactly how much the keyboard
        // covers, then translate the grid up so the rear row clears it.
        var inputPane = TopLevel.GetTopLevel(this)?.InputPane;
        if (inputPane is null) return;

        var occluded = inputPane.OccludedRect;
        if (occluded.Height <= 0) return;

        var rlBottom = RLCorner.TranslatePoint(
            new Point(0, RLCorner.Bounds.Height), this);
        if (!rlBottom.HasValue) return;

        double visibleBottom = Bounds.Height - occluded.Height;
        double shift = rlBottom.Value.Y - visibleBottom;

        if (shift > 0)
        {
            shift += 16; // padding above keyboard edge
            RootGrid.RenderTransform = TransformOperations.Parse($"translate(0px, -{shift}px)");
            _tabletTranslated = true;
        }
    }
}
