using Avalonia.Controls;
using Avalonia.Input;
using InfernoAnalyzer.Charting;
using ScottPlot;
using ScottPlot.Avalonia;

namespace InfernoAnalyzer.Charting.ScottPlot;

/// <summary>
/// What every ScottPlot surface shares: the spec type check and
/// same-spec guard of <see cref="Render"/>, re-rendering on a theme change,
/// the events, and pointer-to-data conversion. A backend implements
/// <see cref="Build"/> for its own spec type.
/// </summary>
/// <typeparam name="TSpec">The spec kind the surface renders.</typeparam>
public abstract class ScottPlotSurface<TSpec> : IChartSurface
    where TSpec : ChartSpec
{
    private ChartTheme _theme = ChartTheme.Dark;

    /// <summary>The spec on screen, or null before the first render.</summary>
    protected TSpec? Spec { get; private set; }

    /// <inheritdoc />
    public abstract Control Control { get; }

    /// <inheritdoc />
    public event Action<double?>? CursorMoved;

    /// <inheritdoc />
    public event Action<ChartPoint>? Clicked;

    /// <inheritdoc />
    public ChartTheme Theme
    {
        get => _theme;
        set
        {
            if (_theme == value)
            {
                return;
            }

            _theme = value;
            if (Spec is { } current)
            {
                Build(current, current);
            }
        }
    }

    /// <inheritdoc />
    public void Render(ChartSpec spec)
    {
        if (spec is not TSpec typed)
        {
            throw new NotSupportedException($"{GetType().Name} cannot render {spec.GetType().Name}");
        }

        if (ReferenceEquals(Spec, typed))
        {
            return;
        }

        var previous = Spec;
        Spec = typed;
        Build(typed, previous);
    }

    /// <summary>
    /// Draw <paramref name="spec"/>. <paramref name="previous"/> is what was
    /// on screen (the same instance when only the theme changed, null on the
    /// first render), for backends that carry view state across renders.
    /// </summary>
    protected abstract void Build(TSpec spec, TSpec? previous);

    /// <inheritdoc />
    public virtual void SetCursor(double? x)
    {
    }

    /// <inheritdoc />
    public abstract void ResetView();

    /// <inheritdoc />
    public virtual void Dispose()
    {
        GC.SuppressFinalize(this);
    }

    /// <summary>Resolve a spec colour for the current theme.</summary>
    protected Color ColorOf(string color) => PlotStyle.Color(_theme, color);

    /// <summary>Raise <see cref="CursorMoved"/>.</summary>
    protected void RaiseCursorMoved(double? x) => CursorMoved?.Invoke(x);

    /// <summary>Raise <see cref="Clicked"/>.</summary>
    protected void RaiseClicked(ChartPoint point) => Clicked?.Invoke(point);

    /// <summary>The data coordinates under the pointer of <paramref name="e"/> on <paramref name="avaPlot"/>.</summary>
    protected static Coordinates ToCoordinates(AvaPlot avaPlot, PointerEventArgs e)
    {
        var position = e.GetPosition(avaPlot);
        var scale = avaPlot.DisplayScale;
        return avaPlot.Plot.GetCoordinates(new Pixel((float)(position.X * scale), (float)(position.Y * scale)));
    }
}

/// <summary>
/// A surface made of a grid of <see cref="AvaPlot"/>s (one per panel): owns
/// the controls and their row/column layout, wires the pointer events once
/// per plot, remembers each plot's limits after a build so
/// <see cref="ResetView"/> can restore them, and releases everything on
/// dispose.
/// </summary>
/// <typeparam name="TSpec">The spec kind the surface renders.</typeparam>
public abstract class ScottPlotGridSurface<TSpec> : ScottPlotSurface<TSpec>
    where TSpec : ChartSpec
{
    private readonly Grid _grid = new();
    private readonly List<Cell> _cells = new();

    /// <inheritdoc />
    public override Control Control => _grid;

    /// <summary>The plot controls in row-major order.</summary>
    protected IReadOnlyList<AvaPlot> PlotControls => _cells.Select(c => c.AvaPlot).ToList();

    /// <summary>The underlying plots in row-major order (also for tests).</summary>
    internal IReadOnlyList<Plot> Plots => _cells.Select(c => c.AvaPlot.Plot).ToList();

    /// <summary>The plot control at <paramref name="index"/>.</summary>
    protected AvaPlot PlotAt(int index) => _cells[index].AvaPlot;

    /// <summary>Number of plots.</summary>
    protected int PlotCount => _cells.Count;

    /// <summary>Create or remove plots so there are <paramref name="count"/> of them, laid out <paramref name="columns"/> per row.</summary>
    protected void EnsurePlots(int count, int columns)
    {
        while (_cells.Count > count)
        {
            var last = _cells[^1];
            Detach(last);
            _grid.Children.Remove(last.AvaPlot);
            _cells.RemoveAt(_cells.Count - 1);
        }

        while (_cells.Count < count)
        {
            var cell = new Cell(new AvaPlot(), _cells.Count);
            Attach(cell);
            _grid.Children.Add(cell.AvaPlot);
            _cells.Add(cell);
            OnPlotCreated(cell.AvaPlot, cell.Index);
        }

        var rows = count == 0 ? 0 : (count + columns - 1) / columns;
        _grid.RowDefinitions.Clear();
        _grid.ColumnDefinitions.Clear();
        for (var r = 0; r < rows; r++)
        {
            _grid.RowDefinitions.Add(new RowDefinition(1, GridUnitType.Star));
        }

        for (var c = 0; c < columns; c++)
        {
            _grid.ColumnDefinitions.Add(new ColumnDefinition(1, GridUnitType.Star));
        }

        for (var i = 0; i < _cells.Count; i++)
        {
            Grid.SetRow(_cells[i].AvaPlot, i / columns);
            Grid.SetColumn(_cells[i].AvaPlot, i % columns);
        }
    }

    /// <summary>Remember every plot's current limits as the view <see cref="ResetView"/> returns to, and draw.</summary>
    protected void CommitView()
    {
        foreach (var cell in _cells)
        {
            PlotStyle.FitFonts(cell.AvaPlot.Plot);
            cell.Home = cell.AvaPlot.Plot.Axes.GetLimits();
            cell.AvaPlot.Refresh();
        }
    }

    /// <inheritdoc />
    public override void ResetView()
    {
        foreach (var cell in _cells)
        {
            if (cell.Home is { } home)
            {
                cell.AvaPlot.Plot.Axes.SetLimits(home);
                cell.AvaPlot.Refresh();
            }
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var cell in _cells)
        {
            Detach(cell);
        }

        _cells.Clear();
        _grid.Children.Clear();
        _grid.RowDefinitions.Clear();
        _grid.ColumnDefinitions.Clear();
        base.Dispose();
    }

    /// <summary>
    /// Choose the pan/zoom gestures of a new plot: x-only over the data area
    /// by default (see <see cref="DataAreaInput"/>); maps override this to
    /// move freely.
    /// </summary>
    protected virtual void InstallInput(AvaPlot avaPlot, int index) => DataAreaInput.Install(avaPlot.UserInputProcessor);

    /// <summary>A plot control was created at <paramref name="index"/>.</summary>
    protected virtual void OnPlotCreated(AvaPlot avaPlot, int index)
    {
    }

    /// <summary>The pointer moved over plot <paramref name="index"/>, at <paramref name="at"/> in data coordinates.</summary>
    protected virtual void OnPointerMoved(int index, Coordinates at, PointerEventArgs e) => RaiseCursorMoved(at.X);

    /// <summary>The pointer left plot <paramref name="index"/>.</summary>
    protected virtual void OnPointerExited(int index) => RaiseCursorMoved(null);

    /// <summary>A button was pressed over plot <paramref name="index"/>, at <paramref name="at"/> in data coordinates.</summary>
    protected virtual void OnPointerPressed(int index, Coordinates at, PointerPressedEventArgs e) =>
        RaiseClicked(new ChartPoint(at.X, at.Y, index));

    private void Attach(Cell cell)
    {
        cell.Moved = (_, e) => OnPointerMoved(cell.Index, ToCoordinates(cell.AvaPlot, e), e);
        cell.Exited = (_, _) => OnPointerExited(cell.Index);
        cell.Pressed = (_, e) => OnPointerPressed(cell.Index, ToCoordinates(cell.AvaPlot, e), e);
        cell.DoubleTapped = (_, _) => ResetView();
        InstallInput(cell.AvaPlot, cell.Index);
        cell.AvaPlot.PointerMoved += cell.Moved;
        cell.AvaPlot.PointerExited += cell.Exited;
        cell.AvaPlot.PointerPressed += cell.Pressed;
        cell.AvaPlot.DoubleTapped += cell.DoubleTapped;
    }

    private static void Detach(Cell cell)
    {
        cell.AvaPlot.PointerMoved -= cell.Moved;
        cell.AvaPlot.PointerExited -= cell.Exited;
        cell.AvaPlot.PointerPressed -= cell.Pressed;
        cell.AvaPlot.DoubleTapped -= cell.DoubleTapped;
    }

    private sealed class Cell
    {
        public Cell(AvaPlot avaPlot, int index)
        {
            AvaPlot = avaPlot;
            Index = index;
        }

        public AvaPlot AvaPlot { get; }
        public int Index { get; }
        public AxisLimits? Home { get; set; }
        public EventHandler<PointerEventArgs>? Moved { get; set; }
        public EventHandler<PointerEventArgs>? Exited { get; set; }
        public EventHandler<PointerPressedEventArgs>? Pressed { get; set; }
        public EventHandler<TappedEventArgs>? DoubleTapped { get; set; }
    }
}
