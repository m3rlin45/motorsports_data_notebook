using TirePressureCalculator.ViewModels;

namespace TirePressureCalculator.Tests;

public class MainViewModelTests
{
    [Fact]
    public void Constructor_CreatesFourCorners()
    {
        var vm = new MainViewModel();

        Assert.NotNull(vm.FrontLeft);
        Assert.NotNull(vm.FrontRight);
        Assert.NotNull(vm.RearLeft);
        Assert.NotNull(vm.RearRight);
    }

    [Fact]
    public void Constructor_CornersHaveCorrectLabels()
    {
        var vm = new MainViewModel();

        Assert.Equal("FL", vm.FrontLeft.Label);
        Assert.Equal("FR", vm.FrontRight.Label);
        Assert.Equal("RL", vm.RearLeft.Label);
        Assert.Equal("RR", vm.RearRight.Label);
    }

    [Fact]
    public void TempAdjustPercent_PropagatesToAllCorners()
    {
        var vm = new MainViewModel();
        vm.TempAdjustPercent = 7.5;

        Assert.Equal(7.5, vm.FrontLeft.TempAdjustPercent);
        Assert.Equal(7.5, vm.FrontRight.TempAdjustPercent);
        Assert.Equal(7.5, vm.RearLeft.TempAdjustPercent);
        Assert.Equal(7.5, vm.RearRight.TempAdjustPercent);
    }

    [Fact]
    public void TempAdjustLabel_Zero_ShowsZero()
    {
        var vm = new MainViewModel();
        vm.TempAdjustPercent = 0;

        Assert.Equal("Temp adjust: 0%", vm.TempAdjustLabel);
    }

    [Fact]
    public void TempAdjustLabel_Positive_ShowsPlus()
    {
        var vm = new MainViewModel();
        vm.TempAdjustPercent = 5.0;

        Assert.Equal("Temp adjust: +5.0%", vm.TempAdjustLabel);
    }

    [Fact]
    public void TempAdjustLabel_Negative_ShowsMinus()
    {
        var vm = new MainViewModel();
        vm.TempAdjustPercent = -3.0;

        Assert.Equal("Temp adjust: -3.0%", vm.TempAdjustLabel);
    }

    [Fact]
    public void ResetCommand_ResetsAllCornersAndSlider()
    {
        var vm = new MainViewModel();
        vm.FrontLeft.CurrentTemp = 35.0;
        vm.FrontLeft.TargetHotTemp = 100.0;
        vm.RearRight.TargetHotPressure = 2.50;
        vm.TempAdjustPercent = 10.0;
        vm.SelectedCondition = "wet";
        vm.LapWithinStint = 12;
        vm.AmbientTempC = 35.0;
        vm.TrackTempC = 42.0;
        vm.CloudCoverPct = 80.0;

        vm.ResetCommand.Execute(null);

        Assert.Equal(0.0, vm.TempAdjustPercent);
        Assert.Equal(20.0, vm.FrontLeft.CurrentTemp);
        Assert.Equal(80.0, vm.FrontLeft.TargetHotTemp);
        Assert.Equal(1.80, vm.RearRight.TargetHotPressure);
        Assert.Equal("dry", vm.SelectedCondition);
        Assert.Equal(5, vm.LapWithinStint);
        Assert.Equal(20.0, vm.AmbientTempC);
        Assert.Null(vm.TrackTempC);
        Assert.Null(vm.CloudCoverPct);
    }

    [Fact]
    public void ResetCommand_CanAlwaysExecute()
    {
        var vm = new MainViewModel();
        Assert.True(vm.ResetCommand.CanExecute(null));
    }
}
