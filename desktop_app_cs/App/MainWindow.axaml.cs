using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App;

/// <summary>The application window; all behaviour lives in <see cref="MainWindowViewModel"/>.</summary>
public partial class MainWindow : Window
{
    /// <summary>Create the window with a fresh view model.</summary>
    public MainWindow()
        : this(new MainWindowViewModel())
    {
    }

    /// <summary>Create the window over <paramref name="viewModel"/>.</summary>
    public MainWindow(MainWindowViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
    }

    /// <summary>The view model.</summary>
    public MainWindowViewModel ViewModel => (MainWindowViewModel)DataContext!;
}
