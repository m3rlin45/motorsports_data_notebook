using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>A compact read-only table (headers bold, numbers right-aligned) for the stats panes.</summary>
public partial class StatsTableView : UserControl
{
    /// <summary>The table to show.</summary>
    public static readonly StyledProperty<StatsTable?> TableProperty =
        AvaloniaProperty.Register<StatsTableView, StatsTable?>(nameof(Table));

    /// <summary>Create the view.</summary>
    public StatsTableView()
    {
        InitializeComponent();
        PropertyChanged += (_, e) =>
        {
            if (e.Property == TableProperty)
            {
                Rebuild();
            }
        };
    }

    /// <summary>The table to show.</summary>
    public StatsTable? Table
    {
        get => GetValue(TableProperty);
        set => SetValue(TableProperty, value);
    }

    private void Rebuild()
    {
        var grid = TableGrid;
        grid.Children.Clear();
        grid.ColumnDefinitions.Clear();
        grid.RowDefinitions.Clear();
        var table = Table;
        TitleText.Text = table?.Title ?? string.Empty;
        if (table is null)
        {
            return;
        }

        for (var c = 0; c < table.Columns.Count; c++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
        }

        grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        for (var c = 0; c < table.Columns.Count; c++)
        {
            grid.Children.Add(Cell(table.Columns[c], 0, c, bold: true, rightAlign: c > 0));
        }

        for (var r = 0; r < table.Rows.Count; r++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            for (var c = 0; c < table.Rows[r].Count; c++)
            {
                grid.Children.Add(Cell(table.Rows[r][c], r + 1, c, bold: false, rightAlign: c > 0));
            }
        }
    }

    private static TextBlock Cell(string text, int row, int column, bool bold, bool rightAlign)
    {
        var block = new TextBlock
        {
            Text = text,
            FontSize = 11,
            FontWeight = bold ? FontWeight.Bold : FontWeight.Normal,
            Margin = new Thickness(6, 1),
            HorizontalAlignment = rightAlign ? HorizontalAlignment.Right : HorizontalAlignment.Left,
        };
        Grid.SetRow(block, row);
        Grid.SetColumn(block, column);
        return block;
    }
}
