using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace TirePressureCalculator.ViewModels;

public class TireCornerViewModel : INotifyPropertyChanged
{
    private string _label = "";
    private double _currentTemp = 20.0;
    private double _targetHotTemp = 80.0;
    private double _targetHotPressure = 1.80;

    public string Label
    {
        get => _label;
        set => SetField(ref _label, value);
    }

    public double CurrentTemp
    {
        get => _currentTemp;
        set { if (SetField(ref _currentTemp, value)) OnPropertyChanged(nameof(ColdPressure)); }
    }

    public double TargetHotTemp
    {
        get => _targetHotTemp;
        set { if (SetField(ref _targetHotTemp, value)) OnPropertyChanged(nameof(ColdPressure)); }
    }

    public double TargetHotPressure
    {
        get => _targetHotPressure;
        set { if (SetField(ref _targetHotPressure, value)) OnPropertyChanged(nameof(ColdPressure)); }
    }

    public double ColdPressure
    {
        get
        {
            double tColdK = _currentTemp + 273.15;
            double tHotK = _targetHotTemp + 273.15;
            if (tHotK <= 0) return 0;
            // Convert gauge pressure to absolute, apply Gay-Lussac's Law, convert back
            double pHotAbs = _targetHotPressure + 1.0;
            double pColdAbs = pHotAbs * (tColdK / tHotK);
            return Math.Round(pColdAbs - 1.0, 3);
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return false;
        field = value;
        OnPropertyChanged(name);
        return true;
    }
}
