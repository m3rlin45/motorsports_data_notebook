using Android.App;
using Android.Content.PM;
using Android.Views;
using Avalonia;
using Avalonia.Android;

namespace TirePressureCalculator.Android;

[Activity(
    Label = "Tire Pressure Calculator",
    Theme = "@style/MyTheme.NoActionBar",
    Icon = "@android:drawable/ic_menu_compass",
    MainLauncher = true,
    WindowSoftInputMode = SoftInput.AdjustResize | SoftInput.StateHidden,
    ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize | ConfigChanges.UiMode | ConfigChanges.Keyboard | ConfigChanges.KeyboardHidden)]
public class MainActivity : AvaloniaMainActivity<App>
{
    protected override AppBuilder CustomizeAppBuilder(AppBuilder builder)
    {
        return base.CustomizeAppBuilder(builder)
            .LogToTrace();
    }

#pragma warning disable CA1422 // Validate platform compatibility (OnBackPressed is obsolete on API 33+)
    public override void OnBackPressed()
    {
        // If the view is zoomed into a quadrant, unzoom instead of closing.
        if (TirePressureCalculator.Views.MainView.Instance?.TryHandleBack() == true)
            return;
        base.OnBackPressed();
    }
#pragma warning restore CA1422
}
