using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Chrome;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.LogicalTree;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Avalonia.Styling;
using Avalonia.VisualTree;
using InfernoAnalyzer.App.Controls;
using InfernoAnalyzer.App.ViewModels;
using InfernoAnalyzer.App.Views;

namespace InfernoAnalyzer.App;

/// <summary>
/// The application window. Behaviour lives in <see cref="MainWindowViewModel"/>;
/// this class is the shell around it that only a window can do: the menu and
/// key bindings, file dialogs and drops, the secondary windows, and saving
/// the window's own state.
/// </summary>
public partial class MainWindow : Window
{
    private readonly TabViewLocator _views;
    private readonly KeyModifiers _cmd;
    private ProfileWindow? _profileWindow;
    private NativeMenu? _recentMenu;
    private NativeMenu? _viewMenu;
    private bool _titleBarHooked;

    /// <summary>Room kept clear at the right of the title row for the minimize, maximize and close buttons.</summary>
    private const double CaptionButtonsWidth = 144;

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
        _views = (TabViewLocator)TabArea.ContentTemplate!;
        _cmd = Application.Current?.PlatformSettings?.HotkeyConfiguration.CommandModifiers ?? KeyModifiers.Control;

        viewModel.SessionA.PickFile = () => PickFileAsync("Open Session A");
        viewModel.SessionB.PickFile = () => PickFileAsync("Open Session B");
        viewModel.ProfileEditorRequested += ShowProfileWindow;
        viewModel.PropertyChanged += OnViewModelPropertyChanged;
        viewModel.RecentFiles.CollectionChanged += OnRecentFilesChanged;
        viewModel.SessionA.PropertyChanged += OnSessionPropertyChanged;
        viewModel.SessionB.PropertyChanged += OnSessionPropertyChanged;

        RestoreWindowState();
        ApplyTheme(viewModel.Theme);
        ArrangeSidebar();
        BuildMenu();
        BuildKeyBindings();
        EnableMica();

        ApplyTitleBar();
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DragLeaveEvent, (_, _) => DropHint.IsVisible = false);
        AddHandler(DragDrop.DropEvent, OnDrop);
        AddHandler(KeyDownEvent, OnTabCommandKeyDown);
        Closing += OnClosing;
        Closed += (_, _) => _views.DisposeViews();
    }

    /// <summary>The view model.</summary>
    public MainWindowViewModel ViewModel => (MainWindowViewModel)DataContext!;

    /// <summary>"Ctrl" or "⌘": the platform's command modifier as shown to the user.</summary>
    private string CmdText => _cmd.HasFlag(KeyModifiers.Meta) ? "⌘" : "Ctrl";

    // ---------------------------------------------------------------- window state

    private void RestoreWindowState()
    {
        var s = ViewModel.Settings;
        // Size and maximized only: a saved position can point at a monitor that is gone, and Wayland ignores it anyway.
        Width = Math.Max(MinWidth, s.WindowWidth);
        Height = Math.Max(MinHeight, s.WindowHeight);
        if (s.Maximized)
        {
            WindowState = WindowState.Maximized;
        }

        Sidebar.Width = Math.Clamp(s.SidebarWidth, Sidebar.MinWidth, Sidebar.MaxWidth);
    }

    private void SaveWindowState()
    {
        var s = ViewModel.Settings;
        s.Maximized = WindowState == WindowState.Maximized;
        if (WindowState == WindowState.Normal)
        {
            s.WindowWidth = Width;
            s.WindowHeight = Height;
        }

        if (!double.IsNaN(Sidebar.Width))
        {
            s.SidebarWidth = Sidebar.Width;
        }

        ViewModel.SaveSettings();
    }

    private void OnClosing(object? sender, WindowClosingEventArgs e) => SaveWindowState();

    /// <summary>
    /// Mica on Windows 11: the page behind the cards takes the desktop tint.
    /// Anywhere it is not available the window keeps its solid page colour.
    /// </summary>
    private void EnableMica()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        TransparencyLevelHint = new[] { WindowTransparencyLevel.Mica };
        TransparencyBackgroundFallback = this.FindResource(ActualThemeVariant, "AppPageBrush") as IBrush ?? Brushes.Gray;
        void Apply()
        {
            if (ActualTransparencyLevel == WindowTransparencyLevel.Mica)
            {
                Background = Brushes.Transparent;
            }
            else
            {
                this[!BackgroundProperty] = this.GetResourceObservable("AppPageBrush").ToBinding();
            }
        }

        PropertyChanged += (_, e) =>
        {
            if (e.Property == ActualTransparencyLevelProperty)
            {
                Apply();
            }
        };
        Apply();
    }

    /// <summary>Whether this platform can host the menu in the title bar (Windows; elsewhere the system draws it).</summary>
    private static bool CanUseTitleBar => OperatingSystem.IsWindows();

    /// <summary>
    /// Windows: extend the client area into the title bar and make the menu
    /// row the title bar: the row is the drag area, the menu inside it stays
    /// interactive, and the right end is left clear for the caption buttons.
    /// Everywhere else, and when the user turns it off, the row is a plain
    /// menu bar under the system's title bar.
    /// </summary>
    private void ApplyTitleBar()
    {
        var wanted = CanUseTitleBar && ViewModel.Settings.MenuInTitleBar;
        ExtendClientAreaToDecorationsHint = wanted;
        void Update()
        {
            var extended = wanted && IsExtendedIntoWindowDecorations;
            WindowDecorationProperties.SetElementRole(TitleRow, extended ? WindowDecorationsElementRole.TitleBar : WindowDecorationsElementRole.None);
            WindowDecorationProperties.SetElementRole(MenuBar, extended ? WindowDecorationsElementRole.User : WindowDecorationsElementRole.None);
            TitleRow.MinHeight = extended ? Math.Max(32, WindowDecorationMargin.Top) : 0;
            TitleIcon.IsVisible = extended;
            TitleText.IsVisible = extended;
            CaptionButtonSpace.Width = extended ? CaptionButtonsWidth : 0;
        }

        if (!_titleBarHooked)
        {
            _titleBarHooked = true;
            PropertyChanged += (_, e) =>
            {
                if (e.Property == IsExtendedIntoWindowDecorationsProperty || e.Property == WindowDecorationMarginProperty)
                {
                    ApplyTitleBar();
                }
            };
        }

        Update();
    }

    private void ToggleTitleBar()
    {
        ViewModel.Settings.MenuInTitleBar = !ViewModel.Settings.MenuInTitleBar;
        ApplyTitleBar();
        BuildViewMenu();
    }

    // ---------------------------------------------------------------- layout

    /// <summary>
    /// A loaded session's panel shares the sidebar's height; an empty one
    /// shrinks to its "Open file" row, so a lone Session A gets the whole
    /// column for its laps.
    /// </summary>
    private void ArrangeSidebar()
    {
        Sidebar.RowDefinitions[0].Height = ViewModel.SessionA.HasSession ? GridLength.Star : GridLength.Auto;
        Sidebar.RowDefinitions[2].Height = ViewModel.SessionB.HasSession ? GridLength.Star : GridLength.Auto;
    }

    private void OnSessionPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(SessionViewModel.HasSession))
        {
            ArrangeSidebar();
        }
    }

    private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(MainWindowViewModel.Theme):
                ApplyTheme(ViewModel.Theme);
                BuildMenu();
                break;
            case nameof(MainWindowViewModel.ActiveTab):
                BuildViewMenu();
                break;
        }
    }

    private static void ApplyTheme(string theme)
    {
        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = theme switch
            {
                "Light" => ThemeVariant.Light,
                "Dark" => ThemeVariant.Dark,
                _ => ThemeVariant.Default,
            };
        }
    }

    // ---------------------------------------------------------------- files

    private async Task<string?> PickFileAsync(string title)
    {
        var folder = ViewModel.Settings.LastFolder is { } last && Directory.Exists(last)
            ? await StorageProvider.TryGetFolderFromPathAsync(last)
            : null;
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            SuggestedStartLocation = folder,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Telemetry files") { Patterns = new[] { "*.xrk", "*.xrz", "*.ibt" } },
                new FilePickerFileType("AIM telemetry") { Patterns = new[] { "*.xrk", "*.xrz" } },
                new FilePickerFileType("iRacing telemetry") { Patterns = new[] { "*.ibt" } },
                FilePickerFileTypes.All,
            },
        });
        return files.FirstOrDefault()?.TryGetLocalPath();
    }

    private static List<string> TelemetryFiles(DragEventArgs e) =>
        (e.DataTransfer?.TryGetFiles() ?? Array.Empty<IStorageItem>())
            .Select(f => f.TryGetLocalPath())
            .Where(MainWindowViewModel.IsTelemetryFile)
            .Select(p => p!)
            .ToList();

    /// <summary>A single file dropped on Session B's panel is the comparison; anywhere else it is Session A.</summary>
    private SessionSlot DropTarget(DragEventArgs e) =>
        e.Source is Visual source && SessionBPanel.IsVisualAncestorOf(source) ? SessionSlot.B : SessionSlot.A;

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        var count = TelemetryFiles(e).Count;
        e.DragEffects = count > 0 ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
        DropHint.IsVisible = count > 0;
        DropHintText.Text = count >= 2 ? "Drop to open as Session A and Session B" : $"Drop to open as Session {DropTarget(e)}";
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        DropHint.IsVisible = false;
        var files = TelemetryFiles(e);
        if (files.Count == 0)
        {
            return;
        }

        e.Handled = true;
        await ViewModel.OpenFilesAsync(files, DropTarget(e));
    }

    // ---------------------------------------------------------------- commands on the visible charts

    private IEnumerable<ChartHost> VisibleCharts() =>
        _views.ViewOf(ViewModel.ActiveTab)?.GetLogicalDescendants().OfType<ChartHost>().Where(h => h.IsEffectivelyVisible && h.Spec is not null)
        ?? Enumerable.Empty<ChartHost>();

    private void FitCharts()
    {
        foreach (var chart in VisibleCharts())
        {
            chart.ResetView();
        }
    }

    private async Task CopyChartImageAsync()
    {
        if (VisibleCharts().FirstOrDefault() is { } chart)
        {
            await chart.CopyImageAsync();
        }
    }

    private void ToggleStats()
    {
        if (ViewModel.ActiveTab.HasStats)
        {
            ViewModel.ActiveTab.ShowStats = !ViewModel.ActiveTab.ShowStats;
        }
    }

    private void ShowProfileWindow()
    {
        if (_profileWindow is not null)
        {
            _profileWindow.Activate();
            return;
        }

        _profileWindow = new ProfileWindow(ViewModel);
        _profileWindow.Closed += (_, _) => _profileWindow = null;
        _profileWindow.Show(this);
    }

    // ---------------------------------------------------------------- menu and keys

    private KeyGesture Gesture(Key key, bool shift = false) => new(key, shift ? _cmd | KeyModifiers.Shift : _cmd);

    /// <summary>A gesture written with "Cmd" for the platform's command modifier ("Cmd+S", "F9", "Escape").</summary>
    private KeyGesture? ParseGesture(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return null;
        }

        var modifiers = KeyModifiers.None;
        var parts = text.Split('+');
        foreach (var part in parts[..^1])
        {
            modifiers |= part switch
            {
                "Cmd" => _cmd,
                "Shift" => KeyModifiers.Shift,
                "Alt" => KeyModifiers.Alt,
                _ => KeyModifiers.None,
            };
        }

        return Enum.TryParse<Key>(parts[^1], out var key) ? new KeyGesture(key, modifiers) : null;
    }

    private static NativeMenuItem Item(string header, Action action, KeyGesture? gesture = null, bool enabled = true)
    {
        var item = new NativeMenuItem(header) { Gesture = gesture, IsEnabled = enabled };
        item.Click += (_, _) => action();
        return item;
    }

    private void BuildMenu()
    {
        var vm = ViewModel;

        _recentMenu = new NativeMenu();
        var file = new NativeMenu
        {
            Item("_Open Session A…", () => _ = vm.OpenSessionA(), Gesture(Key.O)),
            Item("Open Session _B…", () => _ = vm.OpenSessionB(), Gesture(Key.O, shift: true)),
            new NativeMenuItem("Open _Recent") { Menu = _recentMenu },
            new NativeMenuItemSeparator(),
            Item("_Close Session B", () => vm.SessionB.Close()),
            Item("S_wap A and B", () => vm.SwapSessions()),
            new NativeMenuItemSeparator(),
            Item("Vehicle _Profile…", ShowProfileWindow, Gesture(Key.P)),
            new NativeMenuItemSeparator(),
            Item("E_xit", Close),
        };
        BuildRecentMenu();

        _viewMenu = new NativeMenu();
        BuildViewMenu();

        var help = new NativeMenu
        {
            Item("_Keyboard Shortcuts", () => new ShortcutsWindow(CmdText).Show(this), new KeyGesture(Key.F1)),
            Item("_About Inferno Analyzer", () => new AboutWindow().ShowDialog(this)),
        };

        NativeMenu.SetMenu(this, new NativeMenu
        {
            new NativeMenuItem("_File") { Menu = file },
            new NativeMenuItem("_View") { Menu = _viewMenu },
            new NativeMenuItem("_Help") { Menu = help },
        });
    }

    private void OnRecentFilesChanged(object? sender, NotifyCollectionChangedEventArgs e) => BuildRecentMenu();

    private void BuildRecentMenu()
    {
        if (_recentMenu is null)
        {
            return;
        }

        _recentMenu.Items.Clear();
        foreach (var path in ViewModel.RecentFiles)
        {
            _recentMenu.Items.Add(Item(Path.GetFileName(path).Replace("_", "__"), () => _ = ViewModel.OpenRecent(path)));
        }

        if (_recentMenu.Items.Count == 0)
        {
            _recentMenu.Items.Add(new NativeMenuItem("No recent files") { IsEnabled = false });
        }
    }

    /// <summary>The View menu: the tabs, the active tab's own commands, then the window-level toggles.</summary>
    private void BuildViewMenu()
    {
        if (_viewMenu is null)
        {
            return;
        }

        var vm = ViewModel;
        _viewMenu.Items.Clear();
        for (var i = 0; i < vm.Tabs.Count; i++)
        {
            var tab = vm.Tabs[i];
            var item = Item(tab.Header, () => vm.ActiveTab = tab, i < 9 ? Gesture(Key.D1 + i) : null);
            item.ToggleType = MenuItemToggleType.Radio;
            item.IsChecked = ReferenceEquals(tab, vm.ActiveTab);
            _viewMenu.Items.Add(item);
        }

        if (vm.ActiveTab.Commands.Count > 0)
        {
            _viewMenu.Items.Add(new NativeMenuItemSeparator());
            foreach (var command in vm.ActiveTab.Commands)
            {
                _viewMenu.Items.Add(new NativeMenuItem(command.Header) { Command = command.Command, Gesture = ParseGesture(command.Gesture) });
            }
        }

        _viewMenu.Items.Add(new NativeMenuItemSeparator());
        _viewMenu.Items.Add(Item("_Statistics Pane", ToggleStats, Gesture(Key.T), vm.ActiveTab.HasStats));
        _viewMenu.Items.Add(Item("Session Side_bar", vm.ToggleSidebar, Gesture(Key.B)));
        _viewMenu.Items.Add(Item("_Fit Chart in View", FitCharts, Gesture(Key.D0)));
        _viewMenu.Items.Add(Item("_Copy Chart Image", () => _ = CopyChartImageAsync(), Gesture(Key.C, shift: true)));

        var theme = new NativeMenu();
        foreach (var name in new[] { "System", "Light", "Dark" })
        {
            var item = Item(name, () => vm.Theme = name);
            item.ToggleType = MenuItemToggleType.Radio;
            item.IsChecked = vm.Theme == name;
            theme.Items.Add(item);
        }

        _viewMenu.Items.Add(new NativeMenuItemSeparator());
        _viewMenu.Items.Add(new NativeMenuItem("_Theme") { Menu = theme });
        if (CanUseTitleBar)
        {
            var titleBar = Item("_Menu in Title Bar", ToggleTitleBar);
            titleBar.ToggleType = MenuItemToggleType.CheckBox;
            titleBar.IsChecked = vm.Settings.MenuInTitleBar;
            _viewMenu.Items.Add(titleBar);
        }
    }

    /// <summary>
    /// The accelerators. A NativeMenu shows its gestures but only the macOS
    /// global menu acts on them, so the window binds the same keys itself.
    /// </summary>
    private void BuildKeyBindings()
    {
        var vm = ViewModel;
        void Bind(KeyGesture gesture, Action action) =>
            KeyBindings.Add(new KeyBinding { Gesture = gesture, Command = new CommunityToolkit.Mvvm.Input.RelayCommand(action) });

        Bind(Gesture(Key.O), () => _ = vm.OpenSessionA());
        Bind(Gesture(Key.O, shift: true), () => _ = vm.OpenSessionB());
        Bind(Gesture(Key.P), ShowProfileWindow);
        Bind(Gesture(Key.T), ToggleStats);
        Bind(Gesture(Key.B), vm.ToggleSidebar);
        Bind(Gesture(Key.D0), FitCharts);
        Bind(Gesture(Key.C, shift: true), () => _ = CopyChartImageAsync());
        Bind(new KeyGesture(Key.F1), () => new ShortcutsWindow(CmdText).Show(this));
        for (var i = 0; i < Math.Min(vm.Tabs.Count, 9); i++)
        {
            var tab = vm.Tabs[i];
            Bind(Gesture(Key.D1 + i), () => vm.ActiveTab = tab);
        }
    }

    /// <summary>The active tab's own shortcuts (F9, Ctrl+F, Ctrl+S, …): keys no focused control has used.</summary>
    private void OnTabCommandKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Handled)
        {
            return;
        }

        foreach (var command in ViewModel.ActiveTab.Commands)
        {
            if (ParseGesture(command.Gesture) is { } gesture && gesture.Matches(e) && command.Command.CanExecute(null))
            {
                command.Command.Execute(null);
                e.Handled = true;
                return;
            }
        }
    }
}
