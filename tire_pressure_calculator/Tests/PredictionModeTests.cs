using TirePressureCalculator.Services.Modeling;
using TirePressureCalculator.ViewModels;

namespace TirePressureCalculator.Tests;

/// <summary>
/// Tests for the Circuit Prediction UI mode end-to-end: TireCornerViewModel's
/// optional <see cref="TireCornerViewModel.PredictedHotTempC"/> override of
/// the manual hot temp, MainViewModel's mode + prediction inputs wiring, and
/// AppSettings persistence for the new fields.
/// </summary>
public class PredictionModeTests
{
    private static MainViewModel CreateWithModel() =>
        new(TireModelLoader.LoadEmbedded(typeof(TireModel).Assembly));

    // ---------- TireCornerViewModel.PredictedHotTempC ----------

    [Fact]
    public void Corner_PredictedHotTempC_Null_UsesAdjustedHotTemp()
    {
        var corner = new TireCornerViewModel
        {
            Label = "FL",
            CurrentTemp = 20.0,
            TargetHotTemp = 80.0,
            TargetHotPressure = 1.80,
        };
        Assert.Null(corner.PredictedHotTempC);
        Assert.False(corner.HasPrediction);
        // ColdPressure should match the pre-prediction behavior — using AdjustedHotTemp.
        double expectedManual = (1.80 + 1.0) * (293.15 / 353.15) - 1.0;
        Assert.Equal(Math.Round(expectedManual, 3), corner.ColdPressure);
    }

    [Fact]
    public void Corner_PredictedHotTempC_Set_OverridesGayLussacHotSide()
    {
        var corner = new TireCornerViewModel
        {
            Label = "FL",
            CurrentTemp = 20.0,
            TargetHotTemp = 80.0,  // Manual target — should be ignored
            TargetHotPressure = 1.80,
        };
        corner.PredictedHotTempC = 55.0;
        Assert.True(corner.HasPrediction);
        Assert.Equal(55.0, corner.EffectiveHotTempC);
        // Cold pressure should now invert against 55 °C, not 80 °C.
        double expectedPredict = (1.80 + 1.0) * (293.15 / (55.0 + 273.15)) - 1.0;
        Assert.Equal(Math.Round(expectedPredict, 3), corner.ColdPressure);
    }

    [Fact]
    public void Corner_PredictedHotTempC_ClearedToNull_RestoresManual()
    {
        var corner = new TireCornerViewModel
        {
            Label = "FL",
            CurrentTemp = 20.0,
            TargetHotTemp = 80.0,
            TargetHotPressure = 1.80,
        };
        corner.PredictedHotTempC = 55.0;
        corner.PredictedHotTempC = null;
        Assert.False(corner.HasPrediction);
        Assert.Equal(80.0, corner.EffectiveHotTempC);
        double expectedManual = (1.80 + 1.0) * (293.15 / 353.15) - 1.0;
        Assert.Equal(Math.Round(expectedManual, 3), corner.ColdPressure);
    }

    [Fact]
    public void Corner_PredictedHotTempC_FiresPropertyChangeForColdPressure()
    {
        var corner = new TireCornerViewModel
        {
            Label = "FL",
            CurrentTemp = 20.0,
            TargetHotTemp = 80.0,
            TargetHotPressure = 1.80,
        };
        var changes = new List<string?>();
        corner.PropertyChanged += (_, e) => changes.Add(e.PropertyName);

        corner.PredictedHotTempC = 55.0;

        Assert.Contains(nameof(TireCornerViewModel.PredictedHotTempC), changes);
        Assert.Contains(nameof(TireCornerViewModel.ColdPressure), changes);
        Assert.Contains(nameof(TireCornerViewModel.HasPrediction), changes);
    }

    // ---------- MainViewModel mode + wiring ----------

    [Fact]
    public void Mode_DefaultsToManual()
    {
        // Use the parametric ctor so the test doesn't depend on persisted
        // settings from a previous run.
        var vm = new MainViewModel(tireModel: null);
        Assert.Equal(AppMode.Manual, vm.Mode);
        Assert.True(vm.IsManualMode);
        Assert.False(vm.IsPredictionMode);
    }

    [Fact]
    public void Mode_NoModel_TireModelAvailableIsFalse()
    {
        var vm = new MainViewModel(tireModel: null);
        Assert.False(vm.TireModelAvailable);
    }

    [Fact]
    public void Mode_WithEmbeddedModel_PopulatesAvailableTracksAndCars()
    {
        var vm = CreateWithModel();
        Assert.True(vm.TireModelAvailable);
        Assert.Contains("tsukuba_2000", vm.AvailableTracks);
        Assert.Contains("KK-SII", vm.AvailableCars);
        Assert.Contains(vm.AvailableConditions, o => o.Value == "dry");
        Assert.Contains(vm.AvailableConditions, o => o.Value == "damp");
        Assert.Contains(vm.AvailableConditions, o => o.Value == "wet");
    }

    [Fact]
    public void SwitchingToPredictionMode_PushesPredictedHotTempCToCorners()
    {
        var vm = CreateWithModel();
        // Pick a known-dense bucket so the model returns a real number.
        vm.SelectedTrack = "tsukuba_2000";
        vm.SelectedCar = "KK-SII";
        vm.SelectedCondition = "dry";
        vm.LapWithinStint = 5;
        vm.AmbientTempC = 18.0;
        vm.Mode = AppMode.CircuitPrediction;

        Assert.True(vm.IsPredictionMode);
        Assert.True(vm.FrontLeft.HasPrediction);
        Assert.True(vm.FrontRight.HasPrediction);
        Assert.True(vm.RearLeft.HasPrediction);
        Assert.True(vm.RearRight.HasPrediction);
        // Physical sanity: KK-SII at Tsukuba lap 5 dry should warm above ambient.
        Assert.True(vm.FrontLeft.PredictedHotTempC > vm.AmbientTempC);
    }

    [Fact]
    public void SwitchingBackToManual_ClearsPredictedHotTempC()
    {
        var vm = CreateWithModel();
        vm.SelectedTrack = "tsukuba_2000";
        vm.SelectedCar = "KK-SII";
        vm.Mode = AppMode.CircuitPrediction;
        Assert.True(vm.FrontLeft.HasPrediction);

        vm.Mode = AppMode.Manual;
        Assert.False(vm.FrontLeft.HasPrediction);
        Assert.False(vm.FrontRight.HasPrediction);
        Assert.False(vm.RearLeft.HasPrediction);
        Assert.False(vm.RearRight.HasPrediction);
    }

    [Fact]
    public void PredictionMode_LapChange_UpdatesPredictedHotTemp()
    {
        var vm = CreateWithModel();
        vm.SelectedTrack = "tsukuba_2000";
        vm.SelectedCar = "KK-SII";
        vm.SelectedCondition = "dry";
        vm.AmbientTempC = 18.0;
        vm.Mode = AppMode.CircuitPrediction;

        vm.LapWithinStint = 2;
        var earlyHot = vm.FrontLeft.PredictedHotTempC!.Value;
        vm.LapWithinStint = 10;
        var lateHot = vm.FrontLeft.PredictedHotTempC!.Value;
        // Tires get hotter as the stint progresses.
        Assert.True(lateHot > earlyHot);
    }

    [Fact]
    public void PredictionMode_ConditionChange_ChangesPredictedTemps()
    {
        var vm = CreateWithModel();
        vm.SelectedTrack = "tsukuba_2000";
        vm.SelectedCar = "KK-SII";
        vm.LapWithinStint = 5;
        vm.AmbientTempC = 15.0;
        vm.Mode = AppMode.CircuitPrediction;

        vm.SelectedCondition = "dry";
        var dryHot = vm.FrontLeft.PredictedHotTempC!.Value;
        vm.SelectedCondition = "damp";
        var dampHot = vm.FrontLeft.PredictedHotTempC!.Value;
        // Damp should equilibrate to a different hot temp than dry.
        Assert.NotEqual(dryHot, dampHot);
    }

    [Fact]
    public void PredictionMode_CrossCheck_ColdPressureMatchesPredictorOutput()
    {
        // The MainVM should drive corner.PredictedHotTempC from the same
        // numbers the CircuitPredictor would return when called directly.
        var model = TireModelLoader.LoadEmbedded(typeof(TireModel).Assembly);
        var directPredictor = new TirePressureCalculator.Services.CircuitPredictor(model);

        var vm = new MainViewModel(model);
        vm.SelectedTrack = "tsukuba_2000";
        vm.SelectedCar = "KK-SII";
        vm.SelectedCondition = "dry";
        vm.LapWithinStint = 10;
        vm.AmbientTempC = 15.0;
        vm.FrontLeft.CurrentTemp = 15.0;
        vm.FrontLeft.TargetHotPressure = 1.7;
        vm.Mode = AppMode.CircuitPrediction;

        var direct = directPredictor.Predict(
            track: "tsukuba_2000",
            car: "KK-SII",
            condition: "dry",
            lapWithinStint: 10,
            ambientTempC: 15.0,
            trackTempC: null,
            cloudCoverPct: null,
            corner: "fl",
            targetHotPressureBar: 1.7,
            coldTireTempC: 15.0);

        Assert.Equal(direct.PredictedHotTempC, vm.FrontLeft.PredictedHotTempC!.Value, precision: 6);
        // Corner's cold pressure should agree (rounded to 3 decimals).
        Assert.Equal(Math.Round(direct.ColdPressureBar, 3), vm.FrontLeft.ColdPressure);
    }
}
