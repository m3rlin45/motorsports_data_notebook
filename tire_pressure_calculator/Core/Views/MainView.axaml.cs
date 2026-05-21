using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Platform;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.LogicalTree;
using Avalonia.Media;
using Avalonia.Media.Transformation;
using Avalonia.Threading;
using TirePressureCalculator.Navigation;

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

    private Control? _lastFocusedInput;
    private readonly BackNavigationController _backNav = new();

    private ContentControl? ZoomedQuadrant => _backNav.ZoomedQuadrant as ContentControl;
    private bool IsTabletTranslated => _backNav.IsTabletTranslated;

    // Static reference so the Android Activity can ask us to handle Back.
    public static MainView? Instance { get; private set; }

    public MainView()
    {
        InitializeComponent();
        Instance = this;
        _backNav.ResetZoomAction = ApplyZoomReset;
        _backNav.UntranslateAction = ApplyTranslateReset;
        _backNav.Defer = action => Dispatcher.UIThread.Post(action);
        AddHandler(GotFocusEvent, OnChildGotFocus);
        AddHandler(LostFocusEvent, OnChildLostFocus);
        AddHandler(KeyDownEvent, OnChildKeyDown, RoutingStrategies.Bubble, handledEventsToo: true);
        AddHandler(PointerPressedEvent, OnBackgroundPointerPressed, RoutingStrategies.Bubble, handledEventsToo: true);
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
        // The controller defers side effects so a Back press following this
        // event synchronously can still consume the zoom state via
        // TryHandleBack instead of closing the activity.
        _backNav.OnKeyboardClosed();
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

        if (IsPhoneSize && quadrant != ZoomedQuadrant)
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
            else if (IsTabletTranslated)
            {
                ApplyTranslateReset();
                _backNav.NotifyTabletUntranslated();
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
                if (ZoomedQuadrant is not null)
                {
                    ApplyZoomReset();
                    _backNav.NotifyUnzoomed();
                }
                if (IsTabletTranslated)
                {
                    ApplyTranslateReset();
                    _backNav.NotifyTabletUntranslated();
                }
            }
            else if (IsPhoneSize && quadrant != ZoomedQuadrant)
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
                else if (IsTabletTranslated)
                {
                    ApplyTranslateReset();
                    _backNav.NotifyTabletUntranslated();
                }
            }
        });
    }

    private void OnBackgroundPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        // Tap outside any text input dismisses focus, which chains into
        // OnChildLostFocus and reverses the zoom/translate.
        if (FindAncestorOfType<TextBox>(e.Source as Control) is null)
        {
            TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
        }
    }

    private void OnChildKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key != Key.Enter && e.Key != Key.Return) return;

        // Advance focus to the next field on Enter — what every form on every
        // platform does. Falls back to dismissing focus (which chains into
        // OnChildLostFocus and reverses zoom/translate) when there's no next.
        var focused = TopLevel.GetTopLevel(this)?.FocusManager?.GetFocusedElement();
        if (focused is InputElement el)
        {
            var next = KeyboardNavigationHandler.GetNext(el, NavigationDirection.Next);
            if (next is not null && !ReferenceEquals(next, el))
            {
                next.Focus(NavigationMethod.Tab);
                e.Handled = true;
                return;
            }
        }
        TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
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

    internal void ZoomTo(ContentControl quadrant)
    {
        var gridBounds = RootGrid.Bounds;
        if (gridBounds.Width <= 0 || gridBounds.Height <= 0) return;

        // Compute the quadrant's top-left position in grid coordinates.
        // After layout re-runs at scaled size, scroll to this point so the
        // selected quadrant sits at the viewport origin.
        var topLeft = quadrant.TranslatePoint(new Point(0, 0), RootGrid)
                      ?? new Point(0, 0);

        // Pin the RootGrid to the current viewport dimensions so its natural
        // measure is definite. The enclosing LayoutTransformControl will
        // then scale that definite size to 2x.
        RootGrid.Width = gridBounds.Width;
        RootGrid.Height = gridBounds.Height;

        // Enable horizontal scrolling so the ScrollViewer doesn't constrain
        // the scaled-up content width back down to the viewport.
        RootScroll.HorizontalScrollBarVisibility = ScrollBarVisibility.Hidden;

        // Use LayoutTransform (via LayoutTransformControl) rather than
        // RenderTransform so the scale participates in Measure/Arrange.
        // This ensures TextBox selection handles — which are drawn by
        // adorners positioned via the normal layout coordinate system —
        // follow the zoom instead of floating at pre-scale coordinates.
        RootLayoutTransform.LayoutTransform = new ScaleTransform(
            PhoneZoomScale, PhoneZoomScale);
        _backNav.NotifyZoomed(quadrant);

        // Wait for layout to apply the new extent, then scroll so the
        // chosen quadrant is visible at the top of the viewport.
        Dispatcher.UIThread.Post(() =>
        {
            RootScroll.Offset = new Vector(
                topLeft.X * PhoneZoomScale,
                topLeft.Y * PhoneZoomScale);
        });
    }

    /// <summary>
    /// Called by the Android Activity when the Back button is pressed.
    /// Returns true if the back was consumed (zoom reset), false to
    /// let the system handle it (close the app).
    /// </summary>
    public bool TryHandleBack()
    {
        if (_backNav.TryHandleBack())
        {
            Focus();
            return true;
        }
        return false;
    }

    private void ApplyZoomReset()
    {
        RootLayoutTransform.LayoutTransform = null;
        RootGrid.Width = double.NaN;
        RootGrid.Height = double.NaN;
        RootScroll.HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled;
        RootScroll.Offset = new Vector(0, 0);
    }

    private void ApplyTranslateReset()
    {
        RootGrid.RenderTransform = TransformOperations.Parse("scale(1)");
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
            _backNav.NotifyTabletTranslated();
        }
    }
}
