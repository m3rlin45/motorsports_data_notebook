using Avalonia;
using Avalonia.Media;
using Avalonia.Platform;
using Avalonia.Rendering.SceneGraph;
using Avalonia.Skia;
using ScottPlot;
using ScottPlot.Avalonia;
using SkiaSharp;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// ScottPlot's Avalonia control with its drawing moved to the UI thread.
/// <see cref="AvaPlot"/> hands Avalonia a draw operation that runs the whole
/// of ScottPlot's render (layout, auto-scaling, ticks, the plottables, and
/// its events such as AxisLimitsChanged) when the frame is composed, which
/// on Windows is the compositor's thread. Everything else that touches a
/// plot runs on the UI thread, so that was two threads on one plot, events
/// raised where no control may be touched, and exceptions thrown where
/// Avalonia only logs them. Here ScottPlot renders in <see cref="Render"/>,
/// on the UI thread, into a recorded picture, and the draw operation only
/// replays that picture: the compositor's thread never runs ScottPlot, and a
/// failure in a chart surfaces like any other UI-thread exception.
/// </summary>
public sealed class UiThreadPlot : AvaPlot
{
    /// <inheritdoc />
    public override void Render(DrawingContext context)
    {
        var width = (float)Bounds.Width;
        var height = (float)Bounds.Height;
        if (width <= 0 || height <= 0)
        {
            return;
        }

        using var recorder = new SKPictureRecorder();
        var canvas = recorder.BeginRecording(SKRect.Create(width, height));
        Multiplot.Render(canvas, new global::ScottPlot.PixelRect(0, width, height, 0));
        context.Custom(new PictureDrawOperation(new Rect(Bounds.Size), recorder.EndRecording()));
    }

    /// <summary>Replays a recorded picture; all that is left for the render thread.</summary>
    private sealed class PictureDrawOperation : ICustomDrawOperation
    {
        private readonly SKPicture _picture;

        public PictureDrawOperation(Rect bounds, SKPicture picture)
        {
            Bounds = bounds;
            _picture = picture;
        }

        public Rect Bounds { get; }

        public bool HitTest(Point p) => true;

        public bool Equals(ICustomDrawOperation? other) => ReferenceEquals(this, other);

        public void Dispose() => _picture.Dispose();

        public void Render(ImmediateDrawingContext context)
        {
            if (context.TryGetFeature<ISkiaSharpApiLeaseFeature>() is not { } leaseFeature)
            {
                return;
            }

            using var lease = leaseFeature.Lease();
            var canvas = lease.SkCanvas;

            // A layer of its own, as AvaPlot draws: the picture starts by clearing its canvas.
            using var restore = new SKAutoCanvasRestore(canvas, false);
            canvas.SaveLayer();
            canvas.DrawPicture(_picture);
        }
    }
}
