using ScottPlot;
using ScottPlot.Interactivity;
using ScottPlot.Interactivity.UserActionResponses;
using ScottPlot.Interactivity.UserActions;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// Input responses for a strip chart: dragging or wheeling over the data area
/// only ever moves the x-axis; the y-axis is adjusted by dragging or wheeling
/// over the axis itself. ScottPlot's own <c>LockY</c> flags would also lock
/// the axis drag, so the lock is decided per gesture from where it starts.
/// </summary>
internal static class DataAreaInput
{
    /// <summary>Replace the default pan/zoom responses of <paramref name="processor"/>.</summary>
    public static void Install(UserInputProcessor processor)
    {
        processor.RemoveAll<MouseDragPan>();
        processor.RemoveAll<MouseDragZoom>();
        processor.RemoveAll<MouseWheelZoom>();

        var pan = new MouseDragPan(StandardMouseButtons.Left);
        var zoom = new MouseDragZoom(StandardMouseButtons.Right);
        processor.UserActionResponses.Add(new LockYInDataArea(pan, StandardMouseButtons.Left, locked => pan.LockY = locked));
        processor.UserActionResponses.Add(new LockYInDataArea(zoom, StandardMouseButtons.Right, locked => zoom.LockY = locked));
        processor.UserActionResponses.Add(new XOnlyWheelZoom());
    }

    /// <summary>True when <paramref name="pixel"/> is over the data area rather than an axis.</summary>
    private static bool IsDataArea(IPlotControl control, Pixel pixel)
    {
        var plot = control.GetPlotAtPixel(pixel);
        return plot is not null && plot.GetAxis(pixel) is null;
    }

    /// <summary>
    /// Wraps a drag response and flips its y-lock on every button press
    /// according to where the drag starts.
    /// </summary>
    private sealed class LockYInDataArea : IUserActionResponse
    {
        private readonly IUserActionResponse _inner;
        private readonly MouseButton _button;
        private readonly Action<bool> _setLockY;

        public LockYInDataArea(IUserActionResponse inner, MouseButton button, Action<bool> setLockY)
        {
            _inner = inner;
            _button = button;
            _setLockY = setLockY;
        }

        public ResponseInfo Execute(IPlotControl plotControl, IUserAction userActions, KeyboardState keys)
        {
            if (userActions is IMouseButtonAction { IsPressed: true } press && press.Button == _button)
            {
                _setLockY(IsDataArea(plotControl, press.Pixel));
            }

            return _inner.Execute(plotControl, userActions, keys);
        }

        public void ResetState(IPlotControl plotControl) => _inner.ResetState(plotControl);
    }

    /// <summary>
    /// Mouse wheel: over the data area zooms x only; over an axis zooms that
    /// axis (ScottPlot's default behaviour, delegated).
    /// </summary>
    private sealed class XOnlyWheelZoom : IUserActionResponse
    {
        private const double ZoomFraction = 0.15;
        private readonly MouseWheelZoom _default = new(StandardKeys.Shift, StandardKeys.Control);

        public ResponseInfo Execute(IPlotControl plotControl, IUserAction userActions, KeyboardState keys)
        {
            var fraction = userActions switch
            {
                MouseWheelUp => 1 + ZoomFraction,
                MouseWheelDown => 1 / (1 + ZoomFraction),
                _ => 0,
            };
            if (fraction == 0)
            {
                return ResponseInfo.NoActionRequired;
            }

            var pixel = ((IMouseAction)userActions).Pixel;
            if (!IsDataArea(plotControl, pixel))
            {
                return _default.Execute(plotControl, userActions, keys);
            }

            var plot = plotControl.GetPlotAtPixel(pixel)!;
            MouseAxisManipulation.MouseWheelZoom(plot, fraction, 1, pixel, false);
            return new ResponseInfo { RefreshNeeded = true };
        }

        public void ResetState(IPlotControl plotControl) => _default.ResetState(plotControl);
    }
}
