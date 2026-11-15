using Avalonia.Controls;
using InfernoAnalyzer.App.ViewModels;

namespace InfernoAnalyzer.App.Views;

/// <summary>The vehicle profile editor: every tab's channel configuration on its own page, and Save Profile.</summary>
public partial class ProfileWindow : Window
{
    /// <summary>Create the window (designer).</summary>
    public ProfileWindow()
        : this(new MainWindowViewModel())
    {
    }

    /// <summary>Create the window over the main view model.</summary>
    public ProfileWindow(MainWindowViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
        DialogButtons.Arrange(Buttons);
        CloseButton.Click += (_, _) => Close();
    }
}
