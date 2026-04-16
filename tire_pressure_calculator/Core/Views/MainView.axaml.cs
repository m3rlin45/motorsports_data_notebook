using Avalonia;
using Avalonia.Controls;

namespace TirePressureCalculator.Views;

public partial class MainView : UserControl
{
    public static readonly StyledProperty<bool> IsNarrowProperty =
        AvaloniaProperty.Register<MainView, bool>(nameof(IsNarrow));

    // True when the view's aspect ratio is phone-like (taller than wide).
    // The corner template stacks fields vertically at full width in that case,
    // and uses the side-by-side desktop layout otherwise. Works for phones,
    // tablets, and foldables — reflow is driven by bounds, not platform.
    public bool IsNarrow
    {
        get => GetValue(IsNarrowProperty);
        private set => SetValue(IsNarrowProperty, value);
    }

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
            IsNarrow = b.Height > b.Width;
        }
    }
}
