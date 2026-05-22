using System.ComponentModel;
using System.Runtime.CompilerServices;
using TirePressureCalculator.Services.Modeling;

namespace TirePressureCalculator.ViewModels;

public class TireCornerViewModel : INotifyPropertyChanged
{
    private string _label = "";
    private double _currentTemp = 20.0;
    private double _targetHotTemp = 80.0;
    private double _targetHotPressure = 1.80;
    private double _tempAdjustPercent;
    private double? _predictedHotTempC;

    /// <summary>
    /// When non-null, <see cref="ColdPressure"/> uses this value as T_hot
    /// instead of <see cref="AdjustedHotTemp"/>. Set by MainViewModel when
    /// in Circuit Prediction mode; null in Manual mode (preserves existing
    /// behavior).
    /// </summary>
    public double? PredictedHotTempC
    {
        get => _predictedHotTempC;
        set
        {
            if (_predictedHotTempC == value) return;
            _predictedHotTempC = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(EffectiveHotTempC));
            OnPropertyChanged(nameof(PredictedHotTempDisplay));
            OnPropertyChanged(nameof(HasPrediction));
            OnPropertyChanged(nameof(ColdPressure));
        }
    }

    public bool HasPrediction => _predictedHotTempC is not null;

    /// <summary>Hot temperature used by the Gay-Lussac inversion. Equals
    /// <see cref="PredictedHotTempC"/> when set; otherwise <see cref="AdjustedHotTemp"/>.</summary>
    public double EffectiveHotTempC => _predictedHotTempC ?? AdjustedHotTemp;

    public string PredictedHotTempDisplay => _predictedHotTempC is double t
        ? $"Predicted hot: {t:F1} °C"
        : "";

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
        set
        {
            if (SetField(ref _targetHotTemp, value))
            {
                OnPropertyChanged(nameof(AdjustedHotTemp));
                OnPropertyChanged(nameof(TargetHotTempDisplay));
                OnPropertyChanged(nameof(ColdPressure));
            }
        }
    }

    public double TargetHotPressure
    {
        get => _targetHotPressure;
        set { if (SetField(ref _targetHotPressure, value)) OnPropertyChanged(nameof(ColdPressure)); }
    }

    public double TempAdjustPercent
    {
        get => _tempAdjustPercent;
        set
        {
            if (SetField(ref _tempAdjustPercent, value))
            {
                OnPropertyChanged(nameof(AdjustedHotTemp));
                OnPropertyChanged(nameof(IsAdjusted));
                OnPropertyChanged(nameof(TargetHotTempDisplay));
                OnPropertyChanged(nameof(ColdPressure));
            }
        }
    }

    public double AdjustedHotTemp
    {
        get
        {
            double baseK = _targetHotTemp + 273.15;
            double adjustedK = baseK * (1.0 + _tempAdjustPercent / 100.0);
            return Math.Round(adjustedK - 273.15, 1);
        }
    }

    public bool IsAdjusted => _tempAdjustPercent != 0.0;

    public string TargetHotTempDisplay => IsAdjusted
        ? $"{_targetHotTemp:F1} ({AdjustedHotTemp:F1})"
        : $"{_targetHotTemp:F1}";

    public double ColdPressure
    {
        get
        {
            double tHotC = EffectiveHotTempC;
            if (tHotC + EnergyBalance.TZeroCToK <= 0) return 0;
            return Math.Round(
                EnergyBalance.GayLussacColdPressureBar(
                    targetHotPressureBar: _targetHotPressure,
                    tHotC: tHotC,
                    tColdC: _currentTemp),
                3);
        }
    }

    public void ResetToDefaults()
    {
        CurrentTemp = 20.0;
        TargetHotTemp = 80.0;
        TargetHotPressure = 1.80;
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
