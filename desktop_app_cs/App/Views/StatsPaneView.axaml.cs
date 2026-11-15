using Avalonia;
using Avalonia.Controls;
using Avalonia.Input.Platform;
using Avalonia.Layout;
using Avalonia.Media;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>
/// The statistics pane of a tab: its tables one under another, each a
/// compact read-only grid (header over a rule, numbers right-aligned in
/// tabular figures) with a button that copies it as tab-separated text.
/// </summary>
public partial class StatsPaneView : UserControl
{
    /// <summary>The tables to show.</summary>
    public static readonly StyledProperty<IReadOnlyList<StatsTable>?> TablesProperty =
        AvaloniaProperty.Register<StatsPaneView, IReadOnlyList<StatsTable>?>(nameof(Tables));

    /// <summary>Create the view.</summary>
    public StatsPaneView()
    {
        InitializeComponent();
    }

    /// <summary>The tables to show.</summary>
    public IReadOnlyList<StatsTable>? Tables
    {
        get => GetValue(TablesProperty);
        set => SetValue(TablesProperty, value);
    }

    /// <inheritdoc />
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == TablesProperty)
        {
            Rebuild();
        }
    }

    private void Rebuild()
    {
        TablesPanel.Children.Clear();
        foreach (var table in Tables ?? Array.Empty<StatsTable>())
        {
            TablesPanel.Children.Add(BuildTable(table));
        }
    }

    private Control BuildTable(StatsTable table)
    {
        var title = new TextBlock { Text = table.Title, VerticalAlignment = VerticalAlignment.Center };
        title.Classes.Add("section");

        var copy = new Button { Content = new Avalonia.Controls.Shapes.Path { Data = (Geometry)this.FindResource("IconCopy")!, Classes = { "icon" } }, Classes = { "icon" } };
        ToolTip.SetTip(copy, "Copy this table (tab-separated, pastes into a spreadsheet)");
        Avalonia.Automation.AutomationProperties.SetName(copy, $"Copy table {table.Title}");
        copy.Click += async (_, _) =>
        {
            if (TopLevel.GetTopLevel(this)?.Clipboard is { } clipboard)
            {
                await clipboard.SetTextAsync(table.ToTsv());
            }
        };

        var header = new DockPanel { Margin = new Thickness(0, 0, 0, 4) };
        DockPanel.SetDock(copy, Dock.Right);
        header.Children.Add(copy);
        header.Children.Add(title);

        var grid = new Grid { HorizontalAlignment = HorizontalAlignment.Left };
        for (var c = 0; c < table.Columns.Count; c++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
        }

        grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        for (var c = 0; c < table.Columns.Count; c++)
        {
            grid.Children.Add(Cell(table.Columns[c], 0, c, header: true));
        }

        var rule = new Border { Height = 1, Margin = new Thickness(0, 2), [!Border.BackgroundProperty] = this.GetResourceObservable("AppCardStrokeBrush").ToBinding() };
        Grid.SetRow(rule, 1);
        Grid.SetColumnSpan(rule, Math.Max(1, table.Columns.Count));
        grid.Children.Add(rule);

        for (var r = 0; r < table.Rows.Count; r++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            for (var c = 0; c < table.Rows[r].Count; c++)
            {
                grid.Children.Add(Cell(table.Rows[r][c], r + 2, c, header: false));
            }
        }

        var panel = new StackPanel();
        panel.Children.Add(header);
        panel.Children.Add(grid);
        return panel;
    }

    private static TextBlock Cell(string text, int row, int column, bool header)
    {
        var block = new TextBlock
        {
            Text = text,
            Margin = new Thickness(column == 0 ? 0 : 12, 2, 0, 2),
            HorizontalAlignment = column > 0 ? HorizontalAlignment.Right : HorizontalAlignment.Left,
        };
        block.Classes.Add(header ? "section" : "numeric");
        Grid.SetRow(block, row);
        Grid.SetColumn(block, column);
        return block;
    }
}
