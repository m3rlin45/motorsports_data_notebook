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
}
