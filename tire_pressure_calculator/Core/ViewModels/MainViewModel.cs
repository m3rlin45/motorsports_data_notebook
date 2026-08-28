using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using TirePressureCalculator.Localization;
using TirePressureCalculator.Services;
using TirePressureCalculator.Services.Modeling;

namespace TirePressureCalculator.ViewModels;

public class MainViewModel : INotifyPropertyChanged
{
    public TireCornerViewModel FrontLeft { get; }
    public TireCornerViewModel FrontRight { get; }
    public TireCornerViewModel RearLeft { get; }
    public TireCornerViewModel RearRight { get; }
    public ICommand ResetCommand { get; }

    private readonly AppSettings _settings;
    private readonly TireModel? _tireModel;
    private readonly CircuitPredictor? _predictor;
    private double _tempAdjustPercent;

    // ---- Mode + prediction inputs ----

    private AppMode _mode;
    private string? _selectedTrack;
    private string? _selectedCar;
    private string _selectedCondition = "dry";
    private int _lapWithinStint = 5;
    private double _ambientTempC = 20.0;
    private double? _trackTempC;
    private double? _cloudCoverPct;
    private double? _targetLapTimeS;
    private string? _selectedCompound;

    public AppMode Mode
    {
        get => _mode;
        set
        {
            if (_mode == value) return;
            _mode = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsManualMode));
            OnPropertyChanged(nameof(IsPredictionMode));
            _settings.Mode = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public bool IsManualMode => _mode == AppMode.Manual;
    public bool IsPredictionMode => _mode == AppMode.CircuitPrediction;

    public ObservableCollection<string> AvailableTracks { get; } = new();
    public ObservableCollection<string> AvailableCars { get; } = new();
    public ObservableCollection<string> AvailableCompounds { get; } = new();
    public bool HasCompounds => AvailableCompounds.Count > 0;
    public IReadOnlyList<ConditionOption> AvailableConditions { get; } = new[]
    {
        new ConditionOption("dry", "ConditionDry"),
        new ConditionOption("damp", "ConditionDamp"),
        new ConditionOption("wet", "ConditionWet"),
    };

    // Localized strings for direct-content XAML bindings. Each is a plain
    // CLR property — compiled bindings handle these cleanly, unlike
    // indexer paths on a non-DataContext source.
    public string T_ModeLabel => Localizer.Instance["ModeLabel"];
    public string T_ModeManual => Localizer.Instance["ModeManual"];
    public string T_ModePrediction => Localizer.Instance["ModePrediction"];
    public string T_ModelUnavailable => Localizer.Instance["ModelUnavailable"];
    public string T_LanguageLabel => Localizer.Instance["LanguageLabel"];
    public string T_Track => Localizer.Instance["Track"];
    public string T_Car => Localizer.Instance["Car"];
    public string T_Condition => Localizer.Instance["Condition"];
    public string T_LapWithinStint => Localizer.Instance["LapWithinStint"];
    public string T_Ambient => Localizer.Instance["Ambient"];
    public string T_CloudCover => Localizer.Instance["CloudCover"];
    public string T_TargetLapTime => Localizer.Instance["TargetLapTime"];
    public string T_Compound => Localizer.Instance["Compound"];
    public string T_ResetButton => Localizer.Instance["ResetButton"];

    // ---- Language picker ----

    public IReadOnlyList<LanguageOption> AvailableLanguages { get; } = new[]
    {
        new LanguageOption("auto", "LanguageAuto"),
        new LanguageOption("en", "LanguageEnglish"),
        new LanguageOption("ja", "LanguageJapanese"),
    };

    private LanguageOption _selectedLanguage = null!;
    public LanguageOption SelectedLanguage
    {
        get => _selectedLanguage;
        set
        {
            if (ReferenceEquals(_selectedLanguage, value) || value is null) return;
            _selectedLanguage = value;
            OnPropertyChanged();
            Localizer.Instance.SetPreference(value.Code);
            _settings.UiLanguage = value.Code;
            _settings.Save();
            OnPropertyChanged(nameof(TempAdjustLabel));
        }
    }

    public string? SelectedTrack
    {
        get => _selectedTrack;
        set
        {
            if (_selectedTrack == value) return;
            _selectedTrack = value;
            OnPropertyChanged();
            _settings.Prediction.Track = value;
            _settings.Save();
            SnapTargetLap();
            RefreshPredictions();
        }
    }

    public string? SelectedCar
    {
        get => _selectedCar;
        set
        {
            if (_selectedCar == value) return;
            _selectedCar = value;
            OnPropertyChanged();
            _settings.Prediction.Car = value;
            _settings.Save();
            RefreshCompounds();
            SnapTargetLap();
            SnapCornerTargets();
            RefreshPredictions();
        }
    }

    public string SelectedCondition
    {
        get => _selectedCondition;
        set
        {
            if (_selectedCondition == value || value is null) return;
            _selectedCondition = value;
            OnPropertyChanged();
            _settings.Prediction.Condition = value;
            _settings.Save();
            SnapTargetLap();
            SnapCornerTargets();
            RefreshPredictions();
        }
    }

    public int LapWithinStint
    {
        get => _lapWithinStint;
        set
        {
            if (_lapWithinStint == value) return;
            _lapWithinStint = value;
            OnPropertyChanged();
            _settings.Prediction.LapWithinStint = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public double AmbientTempC
    {
        get => _ambientTempC;
        set
        {
            if (_ambientTempC == value) return;
            _ambientTempC = value;
            OnPropertyChanged();
            _settings.Prediction.AmbientTempC = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public double? TrackTempC
    {
        get => _trackTempC;
        set
        {
            if (_trackTempC == value) return;
            _trackTempC = value;
            OnPropertyChanged();
            _settings.Prediction.TrackTempC = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public double? CloudCoverPct
    {
        get => _cloudCoverPct;
        set
        {
            if (_cloudCoverPct == value) return;
            _cloudCoverPct = value;
            OnPropertyChanged();
            _settings.Prediction.CloudCoverPct = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public double? TargetLapTimeS
    {
        get => _targetLapTimeS;
        set
        {
            if (_targetLapTimeS == value) return;
            _targetLapTimeS = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TargetLapTimeText));
            _settings.Prediction.TargetLapTimeS = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    /// <summary>Target lap time as text in the form everyone thinks about
    /// lap times: "1:05.2" (sub-minute laps as plain "58.4"). Accepts either
    /// form on input; blank/invalid input snaps back to the selection's
    /// typical lap.</summary>
    public string TargetLapTimeText
    {
        get => FormatLapTime(_targetLapTimeS);
        set
        {
            var parsed = ParseLapTime(value);
            if (parsed is null)
                SnapTargetLap();
            else
                TargetLapTimeS = Math.Min(900.0, Math.Max(20.0, parsed.Value));
            OnPropertyChanged(); // re-render the normalized m:ss form
        }
    }

    public static string FormatLapTime(double? seconds)
    {
        if (seconds is not double v || !double.IsFinite(v) || v <= 0) return "";
        if (v < 60) return v.ToString("0.0", CultureInfo.InvariantCulture);
        var m = (int)(v / 60);
        var s = v - m * 60;
        return $"{m}:{s.ToString("00.0", CultureInfo.InvariantCulture)}";
    }

    public static double? ParseLapTime(string? raw)
    {
        var t = raw?.Trim().Replace(',', '.');
        if (string.IsNullOrEmpty(t)) return null;
        if (t.Contains(':'))
        {
            var parts = t.Split(':', 2);
            if (int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var m)
                && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var s))
                return m * 60 + s;
            return null;
        }
        return double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
            ? v : null;
    }

    /// <summary>Tire compound — a FORCED choice: every run has exactly one
    /// tire on all four corners, so there is no pooled "default" option; an
    /// unset selection snaps to the car's first compound.</summary>
    public string? SelectedCompound
    {
        get => _selectedCompound;
        set
        {
            if (_selectedCompound == value) return;
            _selectedCompound = value;
            OnPropertyChanged();
            _settings.Prediction.Compound = value;
            _settings.Save();
            RefreshPredictions();
        }
    }

    public bool TireModelAvailable => _tireModel is not null;

    // ---- TempAdjust (existing) ----

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
        ? Localizer.Instance["TempAdjustZero"]
        : Localizer.Instance.Format("TempAdjustFormat", _tempAdjustPercent.ToString("+0.0;-0.0"));

    // ---- Construction ----

    public MainViewModel()
        : this(LoadEmbeddedModelSafely())
    {
    }

    /// <summary>
    /// Test-friendly ctor that accepts an explicit model (or null to disable prediction).
    /// </summary>
    public MainViewModel(TireModel? tireModel)
    {
        _settings = AppSettings.Load();
        _tempAdjustPercent = _settings.TempAdjustPercent;
        _tireModel = tireModel;
        _predictor = tireModel is not null ? new CircuitPredictor(tireModel) : null;

        // Apply the persisted language preference before any view binds.
        Localizer.Instance.SetPreference(_settings.UiLanguage);
        _selectedLanguage = AvailableLanguages.FirstOrDefault(l => l.Code == _settings.UiLanguage)
            ?? AvailableLanguages[0];
        // Fire PropertyChanged(string.Empty) on language switch — INPC
        // convention for "every binding on this object should re-evaluate".
        Localizer.Instance.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Localizer.Language))
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(string.Empty));
        };

        FrontLeft = CreateCorner("FL", _settings.FrontLeft);
        FrontRight = CreateCorner("FR", _settings.FrontRight);
        RearLeft = CreateCorner("RL", _settings.RearLeft);
        RearRight = CreateCorner("RR", _settings.RearRight);

        if (_tireModel is not null)
        {
            foreach (var t in _tireModel.AvailableTracks) AvailableTracks.Add(t);
            foreach (var c in _tireModel.AvailableCars) AvailableCars.Add(c);
        }

        // Restore persisted mode + prediction inputs.
        _mode = _tireModel is null ? AppMode.Manual : _settings.Mode;
        _selectedTrack = _settings.Prediction.Track is string t1 && AvailableTracks.Contains(t1) ? t1
            : AvailableTracks.FirstOrDefault();
        // Resolve alias-pooled car names so a car saved before pooling
        // (e.g. "KK-SII" -> "FJ") lands on its pooled entry, not the default.
        var savedCar = _settings.Prediction.Car is string c0 && _tireModel is not null
            ? _tireModel.ResolveCar(c0) : _settings.Prediction.Car;
        _selectedCar = savedCar is string c1 && AvailableCars.Contains(c1) ? c1
            : AvailableCars.FirstOrDefault();
        _selectedCondition = _settings.Prediction.Condition;
        _lapWithinStint = _settings.Prediction.LapWithinStint;
        _ambientTempC = _settings.Prediction.AmbientTempC;
        _trackTempC = _settings.Prediction.TrackTempC;
        // Cloud cover always carries a value (50% = neutral sky).
        _cloudCoverPct = _settings.Prediction.CloudCoverPct ?? 50.0;
        _targetLapTimeS = _settings.Prediction.TargetLapTimeS;
        _selectedCompound = _settings.Prediction.Compound;

        RefreshCompounds();
        if (_targetLapTimeS is null) SnapTargetLap();
        if (!_settings.LoadedFromStorage) SnapCornerTargets();

        ResetCommand = new RelayCommand(() =>
        {
            TempAdjustPercent = 0;
            FrontLeft.ResetToDefaults();
            FrontRight.ResetToDefaults();
            RearLeft.ResetToDefaults();
            RearRight.ResetToDefaults();
            // Prediction inputs (only reachable when the model is loaded).
            // Routed through the public setters so persistence and
            // RefreshPredictions() run exactly as they do on user input.
            SelectedTrack = AvailableTracks.FirstOrDefault();
            SelectedCar = AvailableCars.FirstOrDefault();
            SelectedCondition = "dry";
            LapWithinStint = 5;
            AmbientTempC = 20.0;
            TrackTempC = null;
            CloudCoverPct = 50.0;
            // Forced defaults: first compound, typical lap, car prefills.
            // The Selected* setters above may have early-returned when the
            // value didn't change, so snap explicitly.
            RefreshCompounds();
            SelectedCompound = AvailableCompounds.FirstOrDefault();
            SnapTargetLap();
            SnapCornerTargets();
        });

        RefreshPredictions();
    }

    private static TireModel? LoadEmbeddedModelSafely()
    {
        try
        {
            return TireModelLoader.LoadEmbedded(typeof(MainViewModel).Assembly);
        }
        catch (Exception)
        {
            // Mobile/browser heads may not have the embedded model wired up,
            // and we still want Manual mode to work without it.
            return null;
        }
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
        vm.PropertyChanged += (_, e) =>
        {
            s.CurrentTemp = vm.CurrentTemp;
            s.TargetHotTemp = vm.TargetHotTemp;
            s.TargetHotPressure = vm.TargetHotPressure;
            _settings.Save();
            // Per-corner target hot pressure change should refresh predictions
            // (the predicted T_hot is independent of target hot pressure, but
            // the cold pressure displayed is a function of it — covered by the
            // corner VM's own ColdPressure recompute).
            if (e.PropertyName == nameof(TireCornerViewModel.TargetHotPressure)
                && IsPredictionMode)
            {
                // Nothing extra to do — the corner already re-renders cold pressure
                // because PredictedHotTempC is unchanged.
            }
        };
        return vm;
    }

    /// <summary>Compound choices depend on the selected car; snap an unset
    /// or foreign selection to the car's first compound.</summary>
    private void RefreshCompounds()
    {
        AvailableCompounds.Clear();
        if (_tireModel is not null && _selectedCar is not null)
        {
            foreach (var c in _tireModel.AvailableCompounds(_selectedCar))
                AvailableCompounds.Add(c);
        }
        OnPropertyChanged(nameof(HasCompounds));
        if (AvailableCompounds.Count > 0
            && (_selectedCompound is null || !AvailableCompounds.Contains(_selectedCompound)))
        {
            SelectedCompound = AvailableCompounds[0];
        }
        else if (AvailableCompounds.Count == 0)
        {
            SelectedCompound = null;
        }
    }

    /// <summary>Target lap time always carries a value: the typical lap for
    /// the current (track, car, condition) selection.</summary>
    private void SnapTargetLap()
    {
        if (_tireModel is null || _selectedTrack is null || _selectedCar is null) return;
        var typ = _tireModel.LookupLapTime(_selectedTrack, _selectedCar, _selectedCondition);
        if (typ.ValueSeconds > 0) TargetLapTimeS = Math.Round(typ.ValueSeconds, 1);
    }

    /// <summary>Corner-card prefills follow the car: target hot temp and hot
    /// pressure snap to the model's steady-state medians for the selected
    /// (car, condition). Buckets missing from the artifact keep whatever the
    /// fields currently hold.</summary>
    private void SnapCornerTargets()
    {
        if (_tireModel is null || _selectedCar is null) return;
        foreach (var (corner, vm) in new[]
                 {
                     ("fl", FrontLeft), ("fr", FrontRight),
                     ("rl", RearLeft), ("rr", RearRight),
                 })
        {
            if (_tireModel.LookupCornerDefaults(_selectedCar, corner, _selectedCondition)
                is CornerDefaultsLookup d)
            {
                vm.TargetHotTemp = Math.Round(d.HotTempC, 1);
                vm.TargetHotPressure = Math.Round(d.HotPressureBar, 2);
            }
        }
    }

    /// <summary>
    /// Recompute per-corner predicted hot temps + push into each corner VM
    /// (or clear them in Manual mode so the corner falls back to AdjustedHotTemp).
    /// </summary>
    private void RefreshPredictions()
    {
        if (!IsPredictionMode || _predictor is null
            || _selectedTrack is null || _selectedCar is null)
        {
            FrontLeft.PredictedHotTempC = null;
            FrontRight.PredictedHotTempC = null;
            RearLeft.PredictedHotTempC = null;
            RearRight.PredictedHotTempC = null;
            return;
        }
        foreach (var (corner, vm) in new[]
                 {
                     ("fl", FrontLeft), ("fr", FrontRight),
                     ("rl", RearLeft), ("rr", RearRight),
                 })
        {
            try
            {
                var prediction = _predictor.Predict(
                    track: _selectedTrack,
                    car: _selectedCar,
                    condition: _selectedCondition,
                    lapWithinStint: _lapWithinStint,
                    ambientTempC: _ambientTempC,
                    trackTempC: _trackTempC,
                    cloudCoverPct: _cloudCoverPct,
                    corner: corner,
                    targetHotPressureBar: vm.TargetHotPressure,
                    coldTireTempC: vm.CurrentTemp,
                    targetLapTimeS: _targetLapTimeS,
                    compound: _selectedCompound);
                vm.PredictedHotTempC = prediction.PredictedHotTempC;
            }
            catch (Exception)
            {
                vm.PredictedHotTempC = null;
            }
        }
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

/// <summary>
/// Picker item that pairs a string value (the data key — "dry"/"damp"/"wet")
/// with a localization key for the display label.
/// </summary>
public sealed class ConditionOption : INotifyPropertyChanged
{
    public string Value { get; }
    public string LocalizationKey { get; }
    public string Display => Localizer.Instance[LocalizationKey];

    public ConditionOption(string value, string localizationKey)
    {
        Value = value;
        LocalizationKey = localizationKey;
        Localizer.Instance.PropertyChanged += (_, _) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Display)));
    }

    public event PropertyChangedEventHandler? PropertyChanged;
}

/// <summary>
/// Picker item for the language menu: a stable code ("auto"/"en"/"ja") plus a
/// localization key for the human-readable name.
/// </summary>
public sealed class LanguageOption : INotifyPropertyChanged
{
    public string Code { get; }
    public string LocalizationKey { get; }
    public string Display => Localizer.Instance[LocalizationKey];

    public LanguageOption(string code, string localizationKey)
    {
        Code = code;
        LocalizationKey = localizationKey;
        Localizer.Instance.PropertyChanged += (_, _) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Display)));
    }

    public event PropertyChangedEventHandler? PropertyChanged;
}
