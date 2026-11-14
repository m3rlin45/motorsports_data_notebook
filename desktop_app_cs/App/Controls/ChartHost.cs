using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Layout;
using Avalonia.LogicalTree;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using InfernoAnalyzer.Charting;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// Hosts one chart in a view: binds a <see cref="ChartSpec"/> from the view
/// model to a surface from <see cref="ChartSurfaces"/>, keeps the surface on
/// the app's theme, joins a <see cref="CursorLinkGroup"/> while attached, and
/// shows an empty state instead of a stale chart when there is no spec. Also
/// supplies the chart's context menu (fit, copy and save as image).
/// </summary>
public sealed class ChartHost : Panel, IDisposable
{
    /// <summary>The chart to draw; null shows <see cref="EmptyText"/>.</summary>
    public static readonly StyledProperty<ChartSpec?> SpecProperty =
        AvaloniaProperty.Register<ChartHost, ChartSpec?>(nameof(Spec));

    /// <summary>The linked-crosshair group the chart joins.</summary>
    public static readonly StyledProperty<CursorLinkGroup?> CursorGroupProperty =
        AvaloniaProperty.Register<ChartHost, CursorLinkGroup?>(nameof(CursorGroup));

    /// <summary>Shown centred when there is no chart: what the user should do to get one.</summary>
    public static readonly StyledProperty<string?> EmptyTextProperty =
        AvaloniaProperty.Register<ChartHost, string?>(nameof(EmptyText));

    private readonly TextBlock _empty = new()
    {
        HorizontalAlignment = HorizontalAlignment.Center,
        VerticalAlignment = VerticalAlignment.Center,
        TextAlignment = TextAlignment.Center,
        TextWrapping = TextWrapping.Wrap,
        MaxWidth = 420,
        Margin = new Thickness(16),
    };

    private IChartSurface? _surface;
    private CursorLinkGroup? _joined;
    private Point? _rightPressedAt;
    private bool _attached;

    /// <summary>Create the host.</summary>
    public ChartHost()
    {
        _empty.Classes.Add("subtle");
        _empty.Classes.Add("title");
        Children.Add(_empty);

        var fit = new MenuItem { Header = "_Fit view", InputGesture = new KeyGesture(Key.Home) };
        fit.Click += (_, _) => ResetView();
        var copy = new MenuItem { Header = "_Copy image" };
        copy.Click += async (_, _) => await CopyImageAsync();
        var save = new MenuItem { Header = "_Save image…" };
        save.Click += async (_, _) => await SaveImageAsync();
        ContextMenu = new ContextMenu { Items = { fit, new Separator(), copy, save } };

        // Right-drag zooms the plot; only a right click that did not move opens the menu.
        AddHandler(PointerPressedEvent, OnPointerPressedTunnel, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        AddHandler(ContextRequestedEvent, OnContextRequested, Avalonia.Interactivity.RoutingStrategies.Tunnel);
    }

    /// <summary>The chart to draw; null shows <see cref="EmptyText"/>.</summary>
    public ChartSpec? Spec
    {
        get => GetValue(SpecProperty);
        set => SetValue(SpecProperty, value);
    }

    /// <summary>The linked-crosshair group the chart joins.</summary>
    public CursorLinkGroup? CursorGroup
    {
        get => GetValue(CursorGroupProperty);
        set => SetValue(CursorGroupProperty, value);
    }

    /// <summary>Shown centred when there is no chart.</summary>
    public string? EmptyText
    {
        get => GetValue(EmptyTextProperty);
        set => SetValue(EmptyTextProperty, value);
    }

    /// <summary>The surface, created with the first spec (null before that).</summary>
    public IChartSurface? Surface => _surface;

    /// <summary>Raised when the user clicks the chart.</summary>
    public event Action<ChartPoint>? ChartClicked;

    /// <inheritdoc />
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == SpecProperty)
        {
            ShowSpec();
        }
        else if (change.Property == CursorGroupProperty)
        {
            JoinGroup();
        }
        else if (change.Property == EmptyTextProperty)
        {
            _empty.Text = EmptyText;
        }
    }

    private void ShowSpec()
    {
        var spec = Spec;
        if (spec is null)
        {
            if (_surface is not null)
            {
                _surface.Control.IsVisible = false; // never leave the previous result on screen as if it were current
            }

            _empty.IsVisible = true;
            return;
        }

        if (_surface is null)
        {
            _surface = ChartSurfaces.CreateFor(spec);
            _surface.Theme = ChartSurfaces.ThemeFor(this);
            _surface.Clicked += OnSurfaceClicked;
            Children.Insert(0, _surface.Control);
            JoinGroup();
        }

        _empty.IsVisible = false;
        _surface.Control.IsVisible = true;
        _surface.Render(spec);
    }

    private void OnSurfaceClicked(ChartPoint point) => ChartClicked?.Invoke(point);

    private void JoinGroup()
    {
        var wanted = _attached ? CursorGroup : null;
        if (ReferenceEquals(wanted, _joined) && (_joined is null || _surface is not null))
        {
            return;
        }

        if (_joined is not null && _surface is not null)
        {
            _joined.Remove(_surface);
        }

        _joined = null;
        if (wanted is not null && _surface is not null)
        {
            wanted.Add(_surface);
            _joined = wanted;
        }
    }

    /// <inheritdoc />
    protected override void OnAttachedToLogicalTree(LogicalTreeAttachmentEventArgs e)
    {
        base.OnAttachedToLogicalTree(e);
        _attached = true;
        ActualThemeVariantChanged += OnThemeChanged;
        OnThemeChanged(this, EventArgs.Empty);
        JoinGroup();
    }

    /// <inheritdoc />
    protected override void OnDetachedFromLogicalTree(LogicalTreeAttachmentEventArgs e)
    {
        base.OnDetachedFromLogicalTree(e);
        _attached = false;
        ActualThemeVariantChanged -= OnThemeChanged;
        JoinGroup();
    }

    private void OnThemeChanged(object? sender, EventArgs e)
    {
        if (_surface is not null)
        {
            _surface.Theme = ChartSurfaces.ThemeFor(this);
        }
    }

    /// <summary>Undo the user's pan and zoom.</summary>
    public void ResetView() => _surface?.ResetView();

    private void OnPointerPressedTunnel(object? sender, PointerPressedEventArgs e)
    {
        var point = e.GetCurrentPoint(this);
        _rightPressedAt = point.Properties.IsRightButtonPressed ? point.Position : null;
    }

    private void OnContextRequested(object? sender, ContextRequestedEventArgs e)
    {
        if (Spec is null)
        {
            e.Handled = true;
            return;
        }

        if (_rightPressedAt is { } pressed && e.TryGetPosition(this, out var released)
            && (Math.Abs(released.X - pressed.X) > 4 || Math.Abs(released.Y - pressed.Y) > 4))
        {
            e.Handled = true; // the end of a right-drag zoom, not a request for the menu
        }
    }

    /// <summary>The chart as drawn now, at the screen's pixel density; null when there is nothing to draw.</summary>
    public RenderTargetBitmap? RenderImage()
    {
        if (_surface is null || Spec is null || Bounds.Width < 1 || Bounds.Height < 1)
        {
            return null;
        }

        var scale = TopLevel.GetTopLevel(this)?.RenderScaling ?? 1.0;
        var size = new PixelSize((int)Math.Ceiling(Bounds.Width * scale), (int)Math.Ceiling(Bounds.Height * scale));
        var bitmap = new RenderTargetBitmap(size, new Vector(96 * scale, 96 * scale));
        bitmap.Render(this);
        return bitmap;
    }

    /// <summary>Put the chart on the clipboard as an image.</summary>
    public async Task CopyImageAsync()
    {
        if (TopLevel.GetTopLevel(this)?.Clipboard is { } clipboard && RenderImage() is { } image)
        {
            await clipboard.SetBitmapAsync(image);
        }
    }

    /// <summary>Save the chart as a PNG the user picks.</summary>
    public async Task SaveImageAsync()
    {
        if (TopLevel.GetTopLevel(this) is not { } top || RenderImage() is not { } image)
        {
            return;
        }

        using (image)
        {
            var file = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
            {
                Title = "Save chart image",
                SuggestedFileName = "chart",
                DefaultExtension = "png",
                FileTypeChoices = new[] { FilePickerFileTypes.ImagePng },
            });
            if (file is null)
            {
                return;
            }

            await using var stream = await file.OpenWriteAsync();
            image.Save(stream, new PngBitmapEncoderOptions());
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_surface is null)
        {
            return;
        }

        _joined?.Remove(_surface);
        _joined = null;
        _surface.Clicked -= OnSurfaceClicked;
        Children.Remove(_surface.Control);
        _surface.Dispose();
        _surface = null;
    }
}
