using Avalonia;
using Avalonia.Controls;

namespace InfernoAnalyzer.App.Views;

/// <summary>The keyboard and mouse reference (Help > Keyboard shortcuts, F1).</summary>
public partial class ShortcutsWindow : Window
{
    /// <summary>Create the window (designer).</summary>
    public ShortcutsWindow()
        : this("Ctrl")
    {
    }

    /// <summary>Create the window; <paramref name="cmd"/> is the platform's command modifier as shown to the user ("Ctrl" or "⌘").</summary>
    public ShortcutsWindow(string cmd)
    {
        InitializeComponent();
        CloseButton.Click += (_, _) => Close();
        Section("Sessions");
        Row($"{cmd}+O", "Open a file as Session A");
        Row($"{cmd}+Shift+O", "Open a file as Session B (comparison)");
        Row("Drop files", "One file opens as Session A, two as A and B");
        Row("Space", "Tick the highlighted laps (Shift+click highlights a range)");
        Section("Window");
        Row($"{cmd}+B", "Show or hide the session sidebar");
        Row($"{cmd}+0", "Fit the chart in view");
        Row($"{cmd}+Shift+C", "Copy the chart as an image");
        Section("Strip chart");
        Row("Click", "Pin the cursor; Esc releases it");
        Row("Left / Right", "Step the cursor one sample (Shift: ten)");
        Row("Up / Down", "Zoom in / out about the cursor");
        Row("Drag, wheel", "Pan and zoom the x-axis; over an axis, that axis");
        Row("Home, double-click", "Back to the whole lap");
        Row("F9", "Distance or time on the x-axis");
        Row($"{cmd}+F", "Search channels (Enter ticks the first match)");
    }

    private void Section(string title)
    {
        var block = new TextBlock { Text = title, Margin = new Thickness(0, Rows.Children.Count == 0 ? 0 : 12, 0, 4) };
        block.Classes.Add("section");
        Rows.Children.Add(block);
    }

    private void Row(string keys, string what)
    {
        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("148,*") };
        var key = new TextBlock { Text = keys };
        key.Classes.Add("subtle");
        var description = new TextBlock { Text = what, TextWrapping = Avalonia.Media.TextWrapping.Wrap };
        Grid.SetColumn(description, 1);
        grid.Children.Add(key);
        grid.Children.Add(description);
        Rows.Children.Add(grid);
    }
}
