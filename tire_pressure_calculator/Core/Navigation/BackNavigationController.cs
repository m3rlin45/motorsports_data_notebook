namespace TirePressureCalculator.Navigation;

/// <summary>
/// Tracks phone-mode zoom and tablet-mode translate state, and coordinates
/// Android back-button handling across the soft keyboard and the activity.
/// </summary>
public class BackNavigationController
{
    /// <summary>Invoked when the zoom state is reset (undo the zoom transform).</summary>
    public Action? ResetZoomAction { get; set; }

    /// <summary>Invoked when the tablet translate state is reset.</summary>
    public Action? UntranslateAction { get; set; }

    /// <summary>
    /// Schedules an action to run later. Defaults to synchronous inline
    /// execution.
    /// </summary>
    public Action<Action> Defer { get; set; } = action => action();

    /// <summary>Identity of the currently zoomed quadrant, or null if not zoomed.</summary>
    public object? ZoomedQuadrant { get; private set; }

    /// <summary>Whether the tablet-mode grid is currently translated up.</summary>
    public bool IsTabletTranslated { get; private set; }

    public bool IsZoomed => ZoomedQuadrant is not null;

    public void NotifyZoomed(object quadrant) => ZoomedQuadrant = quadrant;

    public void NotifyUnzoomed() => ZoomedQuadrant = null;

    public void NotifyTabletTranslated() => IsTabletTranslated = true;

    public void NotifyTabletUntranslated() => IsTabletTranslated = false;

    /// <summary>
    /// Called when the Android soft keyboard closes (by Done, Back, or swipe).
    /// Side effects are deferred via <see cref="Defer"/> so that if the close
    /// was triggered by a Back press, the Activity's OnBackPressed — which
    /// can fire synchronously right after the IME close — still observes the
    /// zoom/translate state and consumes the back press via
    /// <see cref="TryHandleBack"/>. Without deferral, the activity's base
    /// OnBackPressed would run instead and close the app.
    /// </summary>
    public void OnKeyboardClosed()
    {
        Defer(() =>
        {
            if (IsTabletTranslated)
            {
                UntranslateAction?.Invoke();
                IsTabletTranslated = false;
            }
            if (ZoomedQuadrant is not null)
            {
                ResetZoomAction?.Invoke();
                ZoomedQuadrant = null;
            }
        });
    }

    /// <summary>
    /// Called from the Activity's OnBackPressed. Returns true if the back
    /// press was consumed (zoom/translate reset); false if the caller should
    /// let the system handle it (close the activity).
    /// </summary>
    public bool TryHandleBack()
    {
        if (ZoomedQuadrant is not null)
        {
            ResetZoomAction?.Invoke();
            ZoomedQuadrant = null;
            return true;
        }
        if (IsTabletTranslated)
        {
            UntranslateAction?.Invoke();
            IsTabletTranslated = false;
            return true;
        }
        return false;
    }
}
