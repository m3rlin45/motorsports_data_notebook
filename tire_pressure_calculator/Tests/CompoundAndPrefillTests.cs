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
        Assert.NotEqual(pooled.KKelvinPerG2, a052.KKelvinPerG2);
        Assert.NotEqual(pooled.KKelvinPerG2, rs71.KKelvinPerG2);
        // One tire on all four corners: a front corner moves too.
        Assert.NotEqual(Predict("fl", "A052").KKelvinPerG2, Predict("fl", null).KKelvinPerG2);
        // Unknown compound falls back to pooled.
        Assert.Equal(pooled.KKelvinPerG2, Predict("rr", "SLICKS9000").KKelvinPerG2);
    }

    // ---------- The raw cars in the picker (#383) ----------

    // The bundled artifact with a KK-SII entry of its own at Tsukuba (dry):
    // the pooled FJ row's heat x 0.8 and lap time + 1 s.
    private static TireModel WithKKSIIEntry()
    {
        var dto = Model.Dto;
        var g2 = dto.G2TypByTrackCarCond.Single(
            r => r.TrackCanonical == "tsukuba_2000" && r.Car == "FJ" && r.Condition == "dry");
        var lap = dto.LapTimeTypByTrackCarCond.Single(
            r => r.TrackCanonical == "tsukuba_2000" && r.Car == "FJ" && r.Condition == "dry");
        var own = g2 with
        {
            Car = "KK-SII",
            G2Typ = g2.G2Typ * 0.8,
            QTypByCorner = g2.QTypByCorner?.ToDictionary(kv => kv.Key, kv => kv.Value * 0.8),
        };
        return new TireModel(dto with
        {
            G2TypByTrackRawCarCond = new[] { own },
            LapTimeTypByTrackRawCarCond = new[] { lap with { Car = "KK-SII", LapTimeTypS = lap.LapTimeTypS + 1.0 } },
            OutlapTypByTrackRawCarCond = Array.Empty<OutlapEntryDto>(),
        });
    }

    [Fact]
    public void Picker_ListsTheRawCarsInPlaceOfTheirPooledCar_OnlyWithRawCarRows()
    {
        Assert.Equal(Model.FittedCars, Model.AvailableCars);
        Assert.Contains("FJ", Model.AvailableCars);
        Assert.Equal("FJ", Model.SelectableCar("KK-SII"));

        var split = WithKKSIIEntry();
        Assert.Equal(new[] { "Inferno 86", "KK-F", "KK-F-V2", "KK-SII" }, split.AvailableCars);
        Assert.Equal("KK-SII", split.SelectableCar("KK-SII"));
        Assert.Equal("FJ", split.SelectableCar("FJ"));
        Assert.Null(split.SelectableCar("Nobody"));
        // the constants and the prefills resolve a raw name to the pooled car
        Assert.Equal(split.AvailableCompounds("FJ"), split.AvailableCompounds("KK-F"));
        Assert.Equal(split.LookupCornerDefaults("FJ", "fl", "dry"), split.LookupCornerDefaults("KK-SII", "fl", "dry"));
        Assert.Equal(split.LookupTargetHotPressure("FJ", "DRY"), split.LookupTargetHotPressure("KK-SII", "DRY"));
        Assert.Equal(split.LookupK("FJ", "fl", "dry"), split.LookupK("KK-SII", "fl", "dry"));
    }

    private void SaveSettings(string car)
    {
        var settings = new AppSettings
        {
            Mode = AppMode.CircuitPrediction,
            Prediction = new PredictionSettings
            {
                Track = "tsukuba_2000", Car = car, Condition = "dry", TargetLapTimeS = 60.0,
                Compound = "DRY",
            },
        };
        settings.Save();
    }

    [Fact]
    public void SavedKKSII_PredictsFromItsOwnRows_WhenTheArtifactHasThem()
    {
        var split = WithKKSIIEntry();
        SaveSettings("KK-SII");
        var vm = new MainViewModel(split);
        Assert.Equal("KK-SII", vm.SelectedCar);
        Assert.DoesNotContain("FJ", vm.AvailableCars);
        var own = new CircuitPredictor(split).Predict(
            track: "tsukuba_2000", car: "KK-SII", condition: "dry", lapWithinStint: vm.LapWithinStint,
            ambientTempC: vm.AmbientTempC, trackTempC: vm.TrackTempC, cloudCoverPct: vm.CloudCoverPct,
            corner: "fl", targetHotPressureBar: vm.FrontLeft.TargetHotPressure,
            coldTireTempC: vm.FrontLeft.CurrentTemp, targetLapTimeS: 60.0, compound: "DRY");
        var pooled = new CircuitPredictor(split).Predict(
            track: "tsukuba_2000", car: "FJ", condition: "dry", lapWithinStint: vm.LapWithinStint,
            ambientTempC: vm.AmbientTempC, trackTempC: vm.TrackTempC, cloudCoverPct: vm.CloudCoverPct,
            corner: "fl", targetHotPressureBar: vm.FrontLeft.TargetHotPressure,
            coldTireTempC: vm.FrontLeft.CurrentTemp, targetLapTimeS: 60.0, compound: "DRY");
        Assert.Equal(own.PredictedHotTempC, vm.FrontLeft.PredictedHotTempC);
        Assert.True(own.PredictedHotTempC < pooled.PredictedHotTempC, "its own, cooler entry");
        Assert.Equal(pooled.KKelvinPerG2, own.KKelvinPerG2);

        // with today's artifact the same saved KK-SII is the FJ, as before
        SaveSettings("KK-SII");
        Assert.Equal("FJ", new MainViewModel(Model).SelectedCar);
    }

    [Fact]
    public void SavedFJ_KeepsWorking_ButIsNotOfferedBesideTheRawCars()
    {
        SaveSettings("FJ");
        var vm = new MainViewModel(WithKKSIIEntry());
        Assert.Equal("FJ", vm.SelectedCar);
        Assert.Equal(new[] { "Inferno 86", "KK-F", "KK-F-V2", "KK-SII", "FJ" }, vm.AvailableCars);
        Assert.NotNull(vm.FrontLeft.PredictedHotTempC);
        // a fresh start lists the raw cars only
        _storage.Contents = null;
        Assert.DoesNotContain("FJ", new MainViewModel(WithKKSIIEntry()).AvailableCars);
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
        // Foreign compound snaps to the new car's first compound. (Thermal
        // parameters are fitted on dry laps only, so the FJ's rain tire has
        // no fitted compound K and is not enumerated — rain inherits dry.)
        Assert.Contains("DRY", vm.AvailableCompounds);
        Assert.DoesNotContain("RE-71RS", vm.AvailableCompounds);
        Assert.Equal(vm.AvailableCompounds[0], vm.SelectedCompound);
    }

    // ---------- Corner prefills ----------

    [Fact]
    public void FreshSettings_PrefillCornerTargetsFromModel()
    {
        var vm = new MainViewModel(Model); // in-memory storage is empty
        var d = Model.LookupCornerDefaults(vm.SelectedCar!, "fl", vm.SelectedCondition);
        Assert.Equal(Math.Round(d!.Value.HotTempC, 1), vm.FrontLeft.TargetHotTemp);
        // The hot-pressure target comes from the team table for the
        // selected (car, compound), not the measured median.
        var target = Model.LookupTargetHotPressure(vm.SelectedCar!, vm.SelectedCompound);
        Assert.NotNull(target);
        Assert.Equal(Math.Round(target!.Value, 2), vm.FrontLeft.TargetHotPressure);
    }

    [Fact]
    public void HotPressureTarget_FollowsCarAndCompound_TeamTable()
    {
        var vm = new MainViewModel(Model);
        vm.SelectedCar = "FJ";
        vm.SelectedCompound = "DRY";
        Assert.Equal(1.5, vm.FrontLeft.TargetHotPressure);
        Assert.Equal(1.5, vm.RearRight.TargetHotPressure);
        vm.SelectedCompound = "WET";
        Assert.Equal(1.7, vm.RearLeft.TargetHotPressure);

        vm.SelectedCar = "Inferno 86";
        vm.SelectedCompound = "A052";
        Assert.Equal(1.8, vm.FrontRight.TargetHotPressure);
        vm.SelectedCompound = "RE-71RS";
        Assert.Equal(2.0, vm.FrontRight.TargetHotPressure);
        vm.SelectedCompound = "A050";
        Assert.Equal(1.5, vm.FrontRight.TargetHotPressure);
        // The hot temp prefill still follows the measured median.
        var d = Model.LookupCornerDefaults("Inferno 86", "fr", vm.SelectedCondition);
        Assert.Equal(Math.Round(d!.Value.HotTempC, 1), vm.FrontRight.TargetHotTemp);
    }

    // ---------- Targets remembered per configuration ----------

    [Fact]
    public void TunedTargets_ComeBack_WhenTheConfigurationIsReopened()
    {
        var vm = new MainViewModel(Model);
        vm.SelectedCar = "Inferno 86";
        vm.SelectedCompound = "A052";
        Assert.Equal(1.8, vm.FrontLeft.TargetHotPressure);
        vm.FrontLeft.TargetHotPressure = 2.0;
        vm.FrontLeft.TargetHotTemp = 95.0;

        vm.SelectedCar = "FJ"; // first open of FJ / DRY / dry: defaults
        Assert.Equal(1.5, vm.FrontLeft.TargetHotPressure);

        vm.SelectedCar = "Inferno 86";
        vm.SelectedCompound = "A052";
        Assert.Equal(2.0, vm.FrontLeft.TargetHotPressure);
        Assert.Equal(95.0, vm.FrontLeft.TargetHotTemp);
        Assert.Equal(1.8, vm.FrontRight.TargetHotPressure); // untouched corners keep their default

        // A different condition is a different configuration: first open -> defaults.
        vm.SelectedCondition = "wet";
        Assert.Equal(1.8, vm.FrontLeft.TargetHotPressure);
        vm.SelectedCondition = "dry";
        Assert.Equal(2.0, vm.FrontLeft.TargetHotPressure);
    }

    [Fact]
    public void TunedTargets_SurviveReload_AndResetClearsThem()
    {
        var vm1 = new MainViewModel(Model);
        vm1.SelectedCar = "Inferno 86";
        vm1.SelectedCompound = "RE-71RS";
        vm1.RearRight.TargetHotPressure = 2.3;
        vm1.SelectedCar = "FJ";

        var vm2 = new MainViewModel(Model); // same in-memory storage
        vm2.SelectedCar = "Inferno 86";
        vm2.SelectedCompound = "RE-71RS";
        Assert.Equal(2.3, vm2.RearRight.TargetHotPressure);

        vm2.ResetCommand.Execute(null);
        vm2.SelectedCar = "Inferno 86";
        vm2.SelectedCompound = "RE-71RS";
        Assert.Equal(2.0, vm2.RearRight.TargetHotPressure);
    }

    [Fact]
    public void LookupTargetHotPressure_UnknownPairOrNoCompound_IsNull()
    {
        Assert.Null(Model.LookupTargetHotPressure("Inferno 86", "NS-2R"));
        Assert.Null(Model.LookupTargetHotPressure("FJ", null));
        Assert.Equal(2.0, Model.LookupTargetHotPressure("Inferno 86", "RE-71RS"));
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
    public void ConditionChange_ResnapsCornerTargets()
    {
        var vm = new MainViewModel(Model);
        vm.SelectedCar = "FJ";
        vm.SelectedCondition = "wet";
        var wet = Model.LookupCornerDefaults("FJ", "fl", "wet");
        Assert.Equal(Math.Round(wet!.Value.HotTempC, 1), vm.FrontLeft.TargetHotTemp);
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
        // Inferno wet crossed the 5-lap minimum (6 laps), so it may now be exact.
        Assert.True(wet!.Value.Source == "exact" || wet.Value.Source.StartsWith("fallback("));
        // Unknown car -> null (caller keeps its static defaults).
        Assert.Null(Model.LookupCornerDefaults("NoSuchCar", "fl", "dry"));
    }
}
