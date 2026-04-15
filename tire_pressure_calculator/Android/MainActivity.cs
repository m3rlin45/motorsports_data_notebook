using Android.App;
using Android.Content.PM;
using Avalonia;
using Avalonia.Android;

namespace TirePressureCalculator.Android;

[Activity(
    Label = "Tire Pressure Calculator",
    Theme = "@style/MyTheme.NoActionBar",
    Icon = "@android:drawable/ic_menu_compass",
    MainLauncher = true,
    ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize | ConfigChanges.UiMode)]
public class MainActivity : AvaloniaMainActivity<App>
{
    protected override AppBuilder CustomizeAppBuilder(AppBuilder builder)
    {
        return base.CustomizeAppBuilder(builder)
            .LogToTrace();
    }
}
