using Avalonia.Controls;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.Tests;

public class CursorLinkGroupTests
{
    /// <summary>A surface that records SetCursor calls and can raise CursorMoved.</summary>
    private sealed class FakeSurface : IChartSurface
    {
        public List<double?> Received { get; } = new();
        public Control Control => throw new NotSupportedException("not needed by these tests");
        public event Action<double?>? CursorMoved;
        public event Action<ChartPoint>? Clicked;

        public void Render(ChartSpec spec) { }

        public void SetCursor(double? x) => Received.Add(x);

        public void Move(double? x) => CursorMoved?.Invoke(x);

        public void Click(ChartPoint p) => Clicked?.Invoke(p);

        public void Dispose() { }
    }

    [Fact]
    public void A_move_on_one_member_is_pushed_to_the_others_but_not_back_to_the_origin()
    {
        var a = new FakeSurface();
        var b = new FakeSurface();
        var c = new FakeSurface();
        using var group = new CursorLinkGroup();
        group.Add(a);
        group.Add(b);
        group.Add(c);

        a.Move(123.0);

        Assert.Empty(a.Received);
        Assert.Equal(new double?[] { 123.0 }, b.Received);
        Assert.Equal(new double?[] { 123.0 }, c.Received);
        Assert.Equal(123.0, group.Cursor);
    }

    [Fact]
    public void Programmatic_SetCursor_reaches_every_member_and_raises_CursorChanged()
    {
        var a = new FakeSurface();
        var b = new FakeSurface();
        using var group = new CursorLinkGroup();
        group.Add(a);
        group.Add(b);
        double? observed = -1;
        group.CursorChanged += x => observed = x;

        group.SetCursor(null);

        Assert.Equal(new double?[] { null }, a.Received);
        Assert.Equal(new double?[] { null }, b.Received);
        Assert.Null(observed);
    }

    [Fact]
    public void Echoed_moves_during_a_broadcast_do_not_recurse()
    {
        // A backend whose SetCursor raises CursorMoved (some do) must not cause
        // an infinite fan-out.
        var echo = new EchoSurface();
        var other = new FakeSurface();
        using var group = new CursorLinkGroup();
        group.Add(echo);
        group.Add(other);

        other.Move(7.0);

        Assert.Equal(1, echo.SetCursorCalls);
        Assert.Empty(other.Received);
    }

    [Fact]
    public void Removed_members_stop_receiving()
    {
        var a = new FakeSurface();
        var b = new FakeSurface();
        using var group = new CursorLinkGroup();
        group.Add(a);
        group.Add(b);
        group.Remove(b);

        a.Move(1.0);

        Assert.Empty(b.Received);
    }

    private sealed class EchoSurface : IChartSurface
    {
        public int SetCursorCalls { get; private set; }
        public Control Control => throw new NotSupportedException();
        public event Action<double?>? CursorMoved;
        public event Action<ChartPoint>? Clicked;

        public void Render(ChartSpec spec) { }

        public void SetCursor(double? x)
        {
            SetCursorCalls++;
            CursorMoved?.Invoke(x);
        }

        public void Dispose() => Clicked?.Invoke(default);
    }
}
