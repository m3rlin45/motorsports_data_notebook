using TirePressureCalculator.Navigation;

namespace TirePressureCalculator.Tests;

public class BackNavigationControllerTests
{
    [Fact]
    public void TryHandleBack_WhenZoomed_ConsumesAndResets()
    {
        var ctrl = new BackNavigationController();
        bool reset = false;
        ctrl.ResetZoomAction = () => reset = true;
        ctrl.NotifyZoomed(new object());

        var consumed = ctrl.TryHandleBack();

        Assert.True(consumed);
        Assert.True(reset);
        Assert.Null(ctrl.ZoomedQuadrant);
    }

    [Fact]
    public void TryHandleBack_WhenIdle_DoesNotConsume()
    {
        var ctrl = new BackNavigationController();

        var consumed = ctrl.TryHandleBack();

        Assert.False(consumed);
    }

    [Fact]
    public void TryHandleBack_WhenTabletTranslated_ConsumesAndUntranslates()
    {
        var ctrl = new BackNavigationController();
        bool untranslated = false;
        ctrl.UntranslateAction = () => untranslated = true;
        ctrl.NotifyTabletTranslated();

        var consumed = ctrl.TryHandleBack();

        Assert.True(consumed);
        Assert.True(untranslated);
        Assert.False(ctrl.IsTabletTranslated);
    }

    // Bug 2 repro: on Android, pressing Back while a TextBox is focused
    // dispatches two near-simultaneous events — the IME closes (firing
    // OnKeyboardClosed synchronously) and then the Activity's
    // OnBackPressed runs. If OnKeyboardClosed eagerly resets the zoom
    // state, OnBackPressed sees nothing to consume and closes the app.
    //
    // Correct behavior: OnKeyboardClosed must defer its side effects so
    // that a back press following synchronously still observes the zoom
    // state and can consume it via TryHandleBack.
    [Fact]
    public void BackPress_AfterKeyboardClose_StillConsumesAndResets()
    {
        var deferred = new Queue<Action>();
        var ctrl = new BackNavigationController
        {
            Defer = action => deferred.Enqueue(action),
        };
        int resetCalls = 0;
        ctrl.ResetZoomAction = () => resetCalls++;
        ctrl.NotifyZoomed(new object());

        // Step 1: IME closes.
        ctrl.OnKeyboardClosed();

        // Deferred side effects must NOT have run yet — the activity's
        // OnBackPressed will fire synchronously after this, and it needs
        // to observe the zoom state to consume the back press.
        Assert.NotNull(ctrl.ZoomedQuadrant);
        Assert.Equal(0, resetCalls);

        // Step 2: Activity.OnBackPressed runs synchronously.
        var consumed = ctrl.TryHandleBack();

        Assert.True(consumed);
        Assert.Equal(1, resetCalls);
        Assert.Null(ctrl.ZoomedQuadrant);

        // Flush deferred actions: the scheduled reset now no-ops because
        // the state has already been cleared by TryHandleBack.
        while (deferred.TryDequeue(out var action)) action();
        Assert.Equal(1, resetCalls);
    }

    [Fact]
    public void KeyboardClose_WithoutBackPress_EventuallyResetsViaDeferredAction()
    {
        // Simulates the IME dismissed via Done or swipe (no back press).
        // The deferred action runs on the next frame and resets zoom.
        var deferred = new Queue<Action>();
        var ctrl = new BackNavigationController
        {
            Defer = action => deferred.Enqueue(action),
        };
        bool reset = false;
        ctrl.ResetZoomAction = () => reset = true;
        ctrl.NotifyZoomed(new object());

        ctrl.OnKeyboardClosed();
        Assert.False(reset);

        while (deferred.TryDequeue(out var action)) action();

        Assert.True(reset);
        Assert.Null(ctrl.ZoomedQuadrant);
    }

    [Fact]
    public void KeyboardClose_WithTabletTranslate_DeferredUntranslate()
    {
        var deferred = new Queue<Action>();
        var ctrl = new BackNavigationController
        {
            Defer = action => deferred.Enqueue(action),
        };
        bool untranslated = false;
        ctrl.UntranslateAction = () => untranslated = true;
        ctrl.NotifyTabletTranslated();

        ctrl.OnKeyboardClosed();
        Assert.False(untranslated);
        Assert.True(ctrl.IsTabletTranslated);

        // Activity back runs first — consumes.
        var consumed = ctrl.TryHandleBack();
        Assert.True(consumed);
        Assert.True(untranslated);
        Assert.False(ctrl.IsTabletTranslated);
    }
}
