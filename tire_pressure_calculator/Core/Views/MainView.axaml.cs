using Avalonia;
using Avalonia.Controls;

namespace TirePressureCalculator.Views;

public partial class MainView : UserControl
{
    // Aspect ratio thresholds for selecting between the three layouts.
    // Narrow (tall):  aspect < NarrowThreshold → stack fields vertically
    // Wide:           aspect > WideThreshold   → put all 3 fields in a row
    // Medium:         in between               → triangle (side-by-side + below)
    private const double NarrowThreshold = 1.0;
    private const double WideThreshold = 1.7;

    public static readonly StyledProperty<bool> IsNarrowProperty =
        AvaloniaProperty.Register<MainView, bool>(nameof(IsNarrow));

    public static readonly StyledProperty<bool> IsWideProperty =
        AvaloniaProperty.Register<MainView, bool>(nameof(IsWide));

    public bool IsNarrow
    {
        get => GetValue(IsNarrowProperty);
        private set => SetValue(IsNarrowProperty, value);
    }

    public bool IsWide
    {
        get => GetValue(IsWideProperty);
        private set => SetValue(IsWideProperty, value);
    }

    // True when neither narrow nor wide — the default triangle layout used
    // for tablets, desktop, and ~4:3 / ~3:2 aspect ratios.
    public bool IsMedium => !IsNarrow && !IsWide;

    public static readonly DirectProperty<MainView, bool> IsMediumProperty =
        AvaloniaProperty.RegisterDirect<MainView, bool>(nameof(IsMedium), o => o.IsMedium);

    public MainView()
    {
        InitializeComponent();
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == BoundsProperty)
        {
            var b = Bounds;
            if (b.Height <= 0) return;
            double aspect = b.Width / b.Height;
            bool wasMedium = IsMedium;
            IsNarrow = aspect < NarrowThreshold;
            IsWide = aspect > WideThreshold;
            if (IsMedium != wasMedium)
            {
                RaisePropertyChanged(IsMediumProperty, wasMedium, IsMedium);
            }
        }
    }
}
