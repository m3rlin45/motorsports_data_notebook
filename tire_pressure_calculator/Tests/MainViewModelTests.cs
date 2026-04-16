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
    public void ResetCommand_ResetsAllCorners()
    {
        var vm = new MainViewModel();
        vm.FrontLeft.CurrentTemp = 35.0;
        vm.FrontLeft.TargetHotTemp = 100.0;
        vm.RearRight.TargetHotPressure = 2.50;

        vm.ResetCommand.Execute(null);

        Assert.Equal(20.0, vm.FrontLeft.CurrentTemp);
        Assert.Equal(80.0, vm.FrontLeft.TargetHotTemp);
        Assert.Equal(1.80, vm.RearRight.TargetHotPressure);
    }

    [Fact]
    public void ResetCommand_CanAlwaysExecute()
    {
        var vm = new MainViewModel();
        Assert.True(vm.ResetCommand.CanExecute(null));
    }
}
