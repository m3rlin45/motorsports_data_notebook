using TirePressureCalculator.ViewModels;

namespace TirePressureCalculator.Tests;

public class TireCornerViewModelTests
{
    private static TireCornerViewModel CreateDefault() => new()
    {
        Label = "FL",
        CurrentTemp = 20.0,
        TargetHotTemp = 80.0,
        TargetHotPressure = 1.80,
    };

    [Fact]
    public void ColdPressure_DefaultValues_CalculatesCorrectly()
    {
        var vm = CreateDefault();

        // Gay-Lussac: pCold = (pHot_abs) * (tCold_K / tHot_K) - 1 atm
        // = (1.80 + 1.0) * ((20 + 273.15) / (80 + 273.15)) - 1.0
        double expected = 2.80 * (293.15 / 353.15) - 1.0;
        Assert.Equal(Math.Round(expected, 3), vm.ColdPressure);
    }

    [Fact]
    public void ColdPressure_SameTemps_EqualsPressure()
    {
        var vm = CreateDefault();
        vm.CurrentTemp = 80.0;
        vm.TargetHotTemp = 80.0;
        vm.TargetHotPressure = 2.00;

        Assert.Equal(2.0, vm.ColdPressure);
    }

    [Fact]
    public void ColdPressure_HigherColdTemp_HigherColdPressure()
    {
        var vm1 = CreateDefault();
        vm1.CurrentTemp = 10.0;

        var vm2 = CreateDefault();
        vm2.CurrentTemp = 30.0;

        Assert.True(vm2.ColdPressure > vm1.ColdPressure);
    }

    [Fact]
    public void ColdPressure_NegativeTemps_Works()
    {
        var vm = CreateDefault();
        vm.CurrentTemp = -10.0;

        double expected = 2.80 * (263.15 / 353.15) - 1.0;
        Assert.Equal(Math.Round(expected, 3), vm.ColdPressure);
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1.8)]
    [InlineData(3.0)]
    public void ColdPressure_VariousPressures_ScalesLinearly(double targetPressure)
    {
        var vm = CreateDefault();
        vm.TargetHotPressure = targetPressure;

        double ratio = 293.15 / 353.15;
        double expected = (targetPressure + 1.0) * ratio - 1.0;
        Assert.Equal(Math.Round(expected, 3), vm.ColdPressure);
    }

    [Fact]
    public void PropertyChanged_Fires_WhenValuesChange()
    {
        var vm = CreateDefault();
        var changedProperties = new List<string>();
        vm.PropertyChanged += (_, e) => changedProperties.Add(e.PropertyName!);

        vm.CurrentTemp = 25.0;

        Assert.Contains("CurrentTemp", changedProperties);
        Assert.Contains("ColdPressure", changedProperties);
    }

    [Fact]
    public void PropertyChanged_DoesNotFire_WhenSameValue()
    {
        var vm = CreateDefault();
        var changedProperties = new List<string>();
        vm.PropertyChanged += (_, e) => changedProperties.Add(e.PropertyName!);

        vm.CurrentTemp = 20.0;

        Assert.Empty(changedProperties);
    }
}
