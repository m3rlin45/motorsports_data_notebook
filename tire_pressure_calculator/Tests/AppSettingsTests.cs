using System.Text.Json;

namespace TirePressureCalculator.Tests;

public class AppSettingsTests
{
    [Fact]
    public void CornerSettings_Defaults_AreSensible()
    {
        var s = new CornerSettings();

        Assert.Equal(20.0, s.CurrentTemp);
        Assert.Equal(80.0, s.TargetHotTemp);
        Assert.Equal(1.80, s.TargetHotPressure);
    }

    [Fact]
    public void AppSettings_Defaults_HasFourCorners()
    {
        var s = new AppSettings();

        Assert.NotNull(s.FrontLeft);
        Assert.NotNull(s.FrontRight);
        Assert.NotNull(s.RearLeft);
        Assert.NotNull(s.RearRight);
    }

    [Fact]
    public void AppSettings_RoundTrip_PreservesValues()
    {
        var original = new AppSettings
        {
            FrontLeft = new CornerSettings { CurrentTemp = 15.0, TargetHotTemp = 90.0, TargetHotPressure = 2.10 },
            FrontRight = new CornerSettings { CurrentTemp = 16.0, TargetHotTemp = 91.0, TargetHotPressure = 2.11 },
            RearLeft = new CornerSettings { CurrentTemp = 17.0, TargetHotTemp = 92.0, TargetHotPressure = 2.12 },
            RearRight = new CornerSettings { CurrentTemp = 18.0, TargetHotTemp = 93.0, TargetHotPressure = 2.13 },
        };

        var json = JsonSerializer.Serialize(original);
        var deserialized = JsonSerializer.Deserialize<AppSettings>(json)!;

        Assert.Equal(15.0, deserialized.FrontLeft.CurrentTemp);
        Assert.Equal(90.0, deserialized.FrontLeft.TargetHotTemp);
        Assert.Equal(2.10, deserialized.FrontLeft.TargetHotPressure);
        Assert.Equal(18.0, deserialized.RearRight.CurrentTemp);
        Assert.Equal(93.0, deserialized.RearRight.TargetHotTemp);
        Assert.Equal(2.13, deserialized.RearRight.TargetHotPressure);
    }
}
