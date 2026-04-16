using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace TirePressureCalculator.ViewModels;

public class MainViewModel : INotifyPropertyChanged
{
    public TireCornerViewModel FrontLeft { get; }
    public TireCornerViewModel FrontRight { get; }
    public TireCornerViewModel RearLeft { get; }
    public TireCornerViewModel RearRight { get; }
    public ICommand ResetCommand { get; }

    private readonly AppSettings _settings;
    private double _tempAdjustPercent;

    public double TempAdjustPercent
    {
        get => _tempAdjustPercent;
        set
        {
            if (_tempAdjustPercent == value) return;
            _tempAdjustPercent = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TempAdjustLabel));
            FrontLeft.TempAdjustPercent = value;
            FrontRight.TempAdjustPercent = value;
            RearLeft.TempAdjustPercent = value;
            RearRight.TempAdjustPercent = value;
            _settings.TempAdjustPercent = value;
            _settings.Save();
        }
    }

    public string TempAdjustLabel => _tempAdjustPercent == 0
        ? "Temp adjust: 0%"
        : $"Temp adjust: {_tempAdjustPercent:+0.0;-0.0}%";

    public MainViewModel()
    {
        _settings = AppSettings.Load();
        _tempAdjustPercent = _settings.TempAdjustPercent;

        FrontLeft = CreateCorner("FL", _settings.FrontLeft);
        FrontRight = CreateCorner("FR", _settings.FrontRight);
        RearLeft = CreateCorner("RL", _settings.RearLeft);
        RearRight = CreateCorner("RR", _settings.RearRight);

        ResetCommand = new RelayCommand(() =>
        {
            TempAdjustPercent = 0;
            FrontLeft.ResetToDefaults();
            FrontRight.ResetToDefaults();
            RearLeft.ResetToDefaults();
            RearRight.ResetToDefaults();
        });
    }

    private TireCornerViewModel CreateCorner(string label, CornerSettings s)
    {
        var vm = new TireCornerViewModel
        {
            Label = label,
            CurrentTemp = s.CurrentTemp,
            TargetHotTemp = s.TargetHotTemp,
            TargetHotPressure = s.TargetHotPressure,
            TempAdjustPercent = _tempAdjustPercent
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

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

public class RelayCommand(Action execute) : ICommand
{
#pragma warning disable CS0067
    public event EventHandler? CanExecuteChanged;
#pragma warning restore CS0067
    public bool CanExecute(object? parameter) => true;
    public void Execute(object? parameter) => execute();
}
