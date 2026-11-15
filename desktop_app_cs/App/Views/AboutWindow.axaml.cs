using System.Reflection;
using Avalonia.Controls;
using InfernoAnalyzer.App.Settings;

namespace InfernoAnalyzer.App.Views;

/// <summary>Name, version and where the app keeps its files.</summary>
public partial class AboutWindow : Window
{
    /// <summary>Create the window.</summary>
    public AboutWindow()
    {
        InitializeComponent();
        var version = typeof(AboutWindow).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            ?? typeof(AboutWindow).Assembly.GetName().Version?.ToString(3)
            ?? "unknown";
        VersionText.Text = $"Version {version.Split('+')[0]}";
        PathsText.Text = $"Profiles, tracks and settings: {Path.GetDirectoryName(SettingsStore.DefaultPath())}";
        CloseButton.Click += (_, _) => Close();
    }
}
