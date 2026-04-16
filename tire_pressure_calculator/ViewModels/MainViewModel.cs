namespace TirePressureCalculator.ViewModels;

public class MainViewModel
{
    public TireCornerViewModel FrontLeft { get; }
    public TireCornerViewModel FrontRight { get; }
    public TireCornerViewModel RearLeft { get; }
    public TireCornerViewModel RearRight { get; }

    private readonly AppSettings _settings;

    public MainViewModel()
    {
        _settings = AppSettings.Load();

        FrontLeft = CreateCorner("FL", _settings.FrontLeft);
        FrontRight = CreateCorner("FR", _settings.FrontRight);
        RearLeft = CreateCorner("RL", _settings.RearLeft);
        RearRight = CreateCorner("RR", _settings.RearRight);
    }

    private TireCornerViewModel CreateCorner(string label, CornerSettings s)
    {
        var vm = new TireCornerViewModel
        {
            Label = label,
            CurrentTemp = s.CurrentTemp,
            TargetHotTemp = s.TargetHotTemp,
            TargetHotPressure = s.TargetHotPressure
        };
        vm.PropertyChanged += (_, _) =>
        {
            s.CurrentTemp = vm.CurrentTemp;
            s.TargetHotTemp = vm.TargetHotTemp;
            s.TargetHotPressure = vm.TargetHotPressure;
            _settings.Save();
        };
        return vm;
    }
}
