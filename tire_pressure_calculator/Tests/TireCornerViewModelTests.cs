using System.Linq;
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
    public void AdjustedHotTemp_NoAdjustment_EqualsTarget()
    {
        var vm = CreateDefault();
        Assert.Equal(80.0, vm.AdjustedHotTemp);
    }

    [Fact]
    public void AdjustedHotTemp_PositiveAdjust_IncreasesTemp()
    {
        var vm = CreateDefault();
        vm.TempAdjustPercent = 5.0;

        // 5% of (80 + 273.15)K = 5% of 353.15K
        double expectedK = 353.15 * 1.05;
        double expected = Math.Round(expectedK - 273.15, 1);
        Assert.Equal(expected, vm.AdjustedHotTemp);
    }

    [Fact]
    public void AdjustedHotTemp_NegativeAdjust_DecreasesTemp()
    {
        var vm = CreateDefault();
        vm.TempAdjustPercent = -5.0;

        double expectedK = 353.15 * 0.95;
        double expected = Math.Round(expectedK - 273.15, 1);
        Assert.Equal(expected, vm.AdjustedHotTemp);
    }

    [Fact]
    public void ColdPressure_WithTempAdjust_UsesAdjustedTemp()
    {
        var vm = CreateDefault();
        double pressureNoAdjust = vm.ColdPressure;

        vm.TempAdjustPercent = 5.0;
        double pressureWithAdjust = vm.ColdPressure;

        // Higher hot temp -> lower cold pressure
        Assert.True(pressureWithAdjust < pressureNoAdjust);
    }

    [Fact]
    public void IsAdjusted_ZeroPercent_ReturnsFalse()
    {
        var vm = CreateDefault();
        Assert.False(vm.IsAdjusted);
    }

    [Fact]
    public void IsAdjusted_NonZeroPercent_ReturnsTrue()
    {
        var vm = CreateDefault();
        vm.TempAdjustPercent = 3.0;
        Assert.True(vm.IsAdjusted);
    }

    [Fact]
    public void TargetHotTempDisplay_NoAdjust_ShowsBaseOnly()
    {
        var vm = CreateDefault();
        Assert.Equal("80.0", vm.TargetHotTempDisplay);
    }

    [Fact]
    public void TargetHotTempDisplay_WithAdjust_ShowsBothValues()
    {
        var vm = CreateDefault();
        vm.TempAdjustPercent = 5.0;

        Assert.Contains("80.0", vm.TargetHotTempDisplay);
        Assert.Contains("(", vm.TargetHotTempDisplay);
    }

    [Fact]
    public void ResetToDefaults_RestoresInitialValues()
    {
        var vm = CreateDefault();
        vm.CurrentTemp = 35.0;
        vm.TargetHotTemp = 95.0;
        vm.TargetHotPressure = 2.50;

        vm.ResetToDefaults();

        Assert.Equal(20.0, vm.CurrentTemp);
        Assert.Equal(80.0, vm.TargetHotTemp);
        Assert.Equal(1.80, vm.TargetHotPressure);
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

        // Filter out PropertyChanged(string.Empty) — the corner VM uses that
        // signal to refresh all bindings when the Localizer's Language flips,
        // and other tests in this assembly run in parallel and may flip it
        // mid-test. The contract under test is "setting CurrentTemp to its
        // current value fires no value-specific PropertyChanged", so only
        // non-empty property names matter here.
        Assert.Empty(changedProperties.Where(p => !string.IsNullOrEmpty(p)));
    }
}
