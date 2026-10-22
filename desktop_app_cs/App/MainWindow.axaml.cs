using Avalonia.Controls;
using Avalonia.Interactivity;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App;

/// <summary>
/// The spike window: a strip chart of the fixture laps with a linked
/// crosshair (values beside it) and a Distance/Time x-axis toggle.
/// </summary>
public partial class MainWindow : Window
{
    private readonly IChartSurface _chart;
    private Fixture? _fixture;
    private XAxisMode _xAxis = XAxisMode.Distance;

    /// <summary>The link group every chart on this window joins; tests drive the cursor through it.</summary>
    public CursorLinkGroup Cursors { get; } = new();

    /// <summary>Create the window and load the fixture.</summary>
    public MainWindow()
        : this(DefaultFixturePath())
    {
    }

    /// <summary>Create the window for the fixture at <paramref name="fixturePath"/>.</summary>
    public MainWindow(string fixturePath)
    {
        InitializeComponent();

        _chart = ChartSurfaces.CreateStripChart();
        Cursors.Add(_chart);
        ChartHost.Content = _chart.Control;

        DistanceRadio.IsCheckedChanged += OnAxisChanged;
        TimeRadio.IsCheckedChanged += OnAxisChanged;

        Load(fixturePath);
    }

    /// <summary>The fixture path: first CLI argument, else <c>fixtures/laps.json</c> next to the exe.</summary>
    public static string DefaultFixturePath()
    {
        var args = Environment.GetCommandLineArgs();
        return args.Length > 1
            ? args[1]
            : Path.Combine(AppContext.BaseDirectory, "fixtures", "laps.json");
    }

    private void Load(string fixturePath)
    {
        try
        {
            _fixture = Fixture.Load(fixturePath);
            FixtureLabel.Text =
                $"{Path.GetFileName(_fixture.Source)} — {_fixture.Laps.Count} laps, "
                + $"{_fixture.Timebase} timebase, {_fixture.SampleCount:N0} samples";
            Render();
        }
        catch (Exception ex)
        {
            FixtureLabel.Text = $"Could not load {fixturePath}: {ex.Message}";
        }
    }

    private void Render()
    {
        if (_fixture is null)
        {
            return;
        }

        _chart.Render(StripChartBuilder.Build(_fixture, _xAxis));
    }

    private void OnAxisChanged(object? sender, RoutedEventArgs e)
    {
        var mode = TimeRadio.IsChecked == true ? XAxisMode.Time : XAxisMode.Distance;
        if (mode == _xAxis)
        {
            return;
        }

        _xAxis = mode;
        Cursors.SetCursor(null);
        Render();
    }

    /// <inheritdoc />
    protected override void OnClosed(EventArgs e)
    {
        Cursors.Dispose();
        _chart.Dispose();
        base.OnClosed(e);
    }
}
