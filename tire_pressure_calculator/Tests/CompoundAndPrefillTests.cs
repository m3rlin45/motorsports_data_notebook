using TirePressureCalculator.Services;
using TirePressureCalculator.Services.Modeling;
using TirePressureCalculator.ViewModels;

namespace TirePressureCalculator.Tests;

/// <summary>
/// Tests for the compound picker (forced per-car choice, one tire on all
/// four corners) and the car-based input prefills: corner target hot
/// temp/pressure, snap-to-typical target lap time (m:ss text form), and
/// the cloud-cover default. Mirrors the web app's behavior in
/// ../web/js/app.js.
/// </summary>
public class CompoundAndPrefillTests : IDisposable
{
    private sealed class InMemoryStorage : ISettingsStorage
    {
        public string? Contents;
        public string? Read() => Contents;
        public void Write(string contents) => Contents = contents;
    }

    private static readonly TireModel Model =
        TireModelLoader.LoadEmbedded(typeof(TireModel).Assembly);

    private readonly ISettingsStorage _originalStorage;
    private readonly InMemoryStorage _storage = new();

    public CompoundAndPrefillTests()
    {
        _originalStorage = AppSettings.Storage;
        AppSettings.Storage = _storage;
    }

    public void Dispose() => AppSettings.Storage = _originalStorage;

    // ---------- CircuitPredictor compound override ----------

    [Fact]
    public void Predict_CompoundK_OverridesPooledK_OnEveryCorner()
    {
        var predictor = new CircuitPredictor(Model);
        CornerPrediction Predict(string corner, string? compound) => predictor.Predict(
            track: "sodegaura", car: "Inferno 86", condition: "dry",
            lapWithinStint: 5, ambientTempC: 22.0, trackTempC: null,
            cloudCoverPct: null, corner: corner, targetHotPressureBar: 1.9,
            compound: compound);

        var pooled = Predict("rr", null);
        var a052 = Predict("rr", "A052");
        var rs71 = Predict("rr", "RE-71RS");
        Assert.True(a052.KKelvinPerG2 < pooled.KKelvinPerG2, "A052 cooler than pooled");
        Assert.True(rs71.KKelvinPerG2 > a052.KKelvinPerG2, "71RS hotter than A052");
        Assert.True(a052.PredictedHotTempC < rs71.PredictedHotTempC);
        Assert.True(a052.ColdPressureBar > rs71.ColdPressureBar);
        // One tire on all four corners: a front corner moves too.
        Assert.NotEqual(Predict("fl", "A052").KKelvinPerG2, Predict("fl", null).KKelvinPerG2);
        // Unknown compound falls back to pooled.
        Assert.Equal(pooled.KKelvinPerG2, Predict("rr", "SLICKS9000").KKelvinPerG2);
    }

    // ---------- Forced compound selection ----------

    [Fact]
    public void Compounds_EnumeratePerCar_AndSelectionIsForced()
    {
        var vm = new MainViewModel(Model);
        vm.SelectedCar = "Inferno 86";
        Assert.True(vm.HasCompounds);
        Assert.Contains("A052", vm.AvailableCompounds);
        Assert.Contains("RE-71RS", vm.AvailableCompounds);
        Assert.Equal(vm.AvailableCompounds[0], vm.SelectedCompound);

        vm.SelectedCompound = "RE-71RS";
        vm.SelectedCar = "FJ";
        // Foreign compound snaps to the new car's first compound.
        Assert.Contains("DRY", vm.AvailableCompounds);
        Assert.Contains("WET", vm.AvailableCompounds);
        Assert.Equal(vm.AvailableCompounds[0], vm.SelectedCompound);
    }

    // ---------- Corner prefills ----------

    [Fact]
    public void FreshSettings_PrefillCornerTargetsFromModel()
    {
        var vm = new MainViewModel(Model); // in-memory storage is empty
        var d = Model.LookupCornerDefaults(vm.SelectedCar!, "fl", vm.SelectedCondition);
        Assert.Equal(Math.Round(d!.Value.HotTempC, 1), vm.FrontLeft.TargetHotTemp);
        Assert.Equal(Math.Round(d!.Value.HotPressureBar, 2), vm.FrontLeft.TargetHotPressure);
    }

    [Fact]
    public void SavedSettings_KeepUserTunedTargets_UntilSelectionChanges()
    {
        var vm1 = new MainViewModel(Model);
        vm1.FrontLeft.TargetHotTemp = 95.0; // persists to storage

        var vm2 = new MainViewModel(Model);
        Assert.Equal(95.0, vm2.FrontLeft.TargetHotTemp); // no snap on load

        vm2.SelectedCar = vm2.SelectedCar == "FJ" ? "Inferno 86" : "FJ";
        var d = Model.LookupCornerDefaults(vm2.SelectedCar!, "fl", vm2.SelectedCondition);
        Assert.Equal(Math.Round(d!.Value.HotTempC, 1), vm2.FrontLeft.TargetHotTemp);
    }

    [Fact]
    public void ConditionChange_ResnapsCornerTargets_WetCoolerThanDry()
    {
        var vm = new MainViewModel(Model);
        vm.SelectedCar = "FJ";
        var dryTemp = vm.FrontLeft.TargetHotTemp;
        vm.SelectedCondition = "wet";
        Assert.True(vm.FrontLeft.TargetHotTemp < dryTemp,
            $"wet prefill {vm.FrontLeft.TargetHotTemp} should be cooler than dry {dryTemp}");
    }

    // ---------- Target lap time: snap + m:ss text ----------

    [Fact]
    public void FreshSettings_SnapTargetLapToTypical_AndCloudTo50()
    {
        var vm = new MainViewModel(Model);
        var typ = Model.LookupLapTime(vm.SelectedTrack!, vm.SelectedCar!, vm.SelectedCondition);
        Assert.Equal(Math.Round(typ.ValueSeconds, 1), vm.TargetLapTimeS);
        Assert.Equal(50.0, vm.CloudCoverPct);
    }

    [Fact]
    public void LapTimeText_FormatsAndParses_MinutesSeconds()
    {
        Assert.Equal("2:09.2", MainViewModel.FormatLapTime(129.24));
        Assert.Equal("1:05.2", MainViewModel.FormatLapTime(65.2));
        Assert.Equal("58.4", MainViewModel.FormatLapTime(58.4));
        Assert.Equal("", MainViewModel.FormatLapTime(null));

        Assert.Equal(65.2, MainViewModel.ParseLapTime("1:05.2")!.Value, 3);
        Assert.Equal(65.2, MainViewModel.ParseLapTime("65.2")!.Value, 3);
        Assert.Equal(65.2, MainViewModel.ParseLapTime("1:05,2")!.Value, 3);
        Assert.Null(MainViewModel.ParseLapTime(""));
        Assert.Null(MainViewModel.ParseLapTime("abc"));
    }

    [Fact]
    public void LapTimeText_BlankInput_SnapsBackToTypical()
    {
        var vm = new MainViewModel(Model);
        vm.TargetLapTimeText = "1:00.5";
        Assert.Equal(60.5, vm.TargetLapTimeS);

        vm.TargetLapTimeText = "";
        var typ = Model.LookupLapTime(vm.SelectedTrack!, vm.SelectedCar!, vm.SelectedCondition);
        Assert.Equal(Math.Round(typ.ValueSeconds, 1), vm.TargetLapTimeS);
    }

    // ---------- Model lookups ----------

    [Fact]
    public void LookupCornerDefaults_ConditionChain_FallsBackForMissingWet()
    {
        // Inferno 86 has no wet prefill bucket (under the 5-lap minimum);
        // the chain falls back toward damp/dry instead of returning null.
        var wet = Model.LookupCornerDefaults("Inferno 86", "fl", "wet");
        Assert.NotNull(wet);
        Assert.StartsWith("fallback(", wet!.Value.Source);
        // Unknown car -> null (caller keeps its static defaults).
        Assert.Null(Model.LookupCornerDefaults("NoSuchCar", "fl", "dry"));
    }
}
