using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TirePressureCalculator.Services.Modeling;

// DTOs mirroring data/tire_dataset/tire_model.json (schema v3; v2
// artifacts still load). See docs/tire_model.md §2.7 for the schema
// reference.

public sealed record TireModelDto(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("fit_at_utc")] string FitAtUtc,
    [property: JsonPropertyName("model_form")] string ModelForm,
    [property: JsonPropertyName("gay_lussac")] GayLussacConfigDto GayLussac,
    [property: JsonPropertyName("energy_balance")] EnergyBalanceConfigDto EnergyBalance,
    [property: JsonPropertyName("conditions")] ConditionsConfigDto Conditions,
    [property: JsonPropertyName("corners")] IReadOnlyList<string> Corners,
    [property: JsonPropertyName("min_samples_per_bucket")] int MinSamplesPerBucket,
    [property: JsonPropertyName("priors_when_no_fit")] PriorsDto PriorsWhenNoFit,
    [property: JsonPropertyName("tau_sec_by_car_corner_cond")] IReadOnlyList<TauEntryDto> TauSecByCarCornerCond,
    [property: JsonPropertyName("K_buckets")] IReadOnlyList<KBucketEntryDto> KBuckets,
    [property: JsonPropertyName("c_track_by_track")] IReadOnlyList<CTrackEntryDto> CTrackByTrack,
    [property: JsonPropertyName("g2_typ_by_track_car_cond")] IReadOnlyList<G2EntryDto> G2TypByTrackCarCond,
    [property: JsonPropertyName("lap_time_typ_by_track_car_cond")] IReadOnlyList<LapTimeEntryDto> LapTimeTypByTrackCarCond,
    [property: JsonPropertyName("g2_lap_time_model")] G2LapTimeModelDto? G2LapTimeModel = null,
    [property: JsonPropertyName("K_by_car_compound_corner_cond")] IReadOnlyList<KCompoundEntryDto>? KByCarCompoundCornerCond = null,
    [property: JsonPropertyName("corner_defaults_by_car_corner_cond")] IReadOnlyList<CornerDefaultsEntryDto>? CornerDefaultsByCarCornerCond = null,
    [property: JsonPropertyName("car_aliases")] IReadOnlyDictionary<string, string>? CarAliases = null
);

// ---- Compound-aware K + UI prefill medians (schema v3 additive) ----

public sealed record KCompoundEntryDto(
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("compound")] string Compound,
    [property: JsonPropertyName("corner")] string Corner,
    [property: JsonPropertyName("condition")] string Condition,
    [property: JsonPropertyName("value_kelvin_per_g2")] double ValueKelvinPerG2,
    [property: JsonPropertyName("stderr_kelvin_per_g2")] double StderrKelvinPerG2,
    [property: JsonPropertyName("n_laps")] int NLaps
);

public sealed record CornerDefaultsEntryDto(
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("corner")] string Corner,
    [property: JsonPropertyName("condition")] string Condition,
    [property: JsonPropertyName("hot_temp_c")] double HotTempC,
    [property: JsonPropertyName("hot_pressure_bar")] double HotPressureBar,
    [property: JsonPropertyName("n_laps_used")] int NLapsUsed
);

// ---- Target-lap-time feature (schema v3) ----

public sealed record G2LapTimeModelDto(
    [property: JsonPropertyName("method")] string? Method,
    [property: JsonPropertyName("formula")] string? Formula,
    [property: JsonPropertyName("default_exponent")] double DefaultExponent,
    [property: JsonPropertyName("multiplier_clamp")] MultiplierClampDto? MultiplierClamp
);

public sealed record MultiplierClampDto(
    [property: JsonPropertyName("min")] double Min,
    [property: JsonPropertyName("max")] double Max
);

/// <summary>Piecewise-linear g² vs lap-time curve (sector-fit, see
/// src/motorsports_data_notebook/tire_model/sectors.py).</summary>
public sealed record G2CurveDto(
    [property: JsonPropertyName("lap_time_s")] IReadOnlyList<double> LapTimeS,
    [property: JsonPropertyName("g2")] IReadOnlyList<double> G2,
    [property: JsonPropertyName("n_laps")] int NLaps
);

public sealed record GayLussacConfigDto(
    [property: JsonPropertyName("p_atm_bar")] double PAtmBar,
    [property: JsonPropertyName("t_zero_c_to_k")] double TZeroCToK,
    [property: JsonPropertyName("t_cold_uses")] string TColdUses
);

public sealed record EnergyBalanceConfigDto(
    [property: JsonPropertyName("w_road")] double WRoad,
    [property: JsonPropertyName("w_road_fitted")] bool WRoadFitted,
    [property: JsonPropertyName("t_road_proxy")] TRoadProxyConfigDto TRoadProxy
);

public sealed record TRoadProxyConfigDto(
    [property: JsonPropertyName("formula")] string Formula,
    [property: JsonPropertyName("delta_sun_max_c")] double DeltaSunMaxC,
    [property: JsonPropertyName("sun_factor_default")] double SunFactorDefault
);

public sealed record ConditionsConfigDto(
    [property: JsonPropertyName("values")] IReadOnlyList<string> Values,
    [property: JsonPropertyName("default")] string Default
);

public sealed record PriorsDto(
    [property: JsonPropertyName("tau_sec_seconds")] double TauSecSeconds,
    [property: JsonPropertyName("K_kelvin_per_g2")] double KKelvinPerG2,
    [property: JsonPropertyName("c_track")] double CTrack
);

public sealed record TauEntryDto(
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("corner")] string Corner,
    [property: JsonPropertyName("condition")] string Condition,
    [property: JsonPropertyName("value_seconds")] double ValueSeconds,
    [property: JsonPropertyName("stderr_seconds")] double StderrSeconds,
    [property: JsonPropertyName("n_samples_used")] int NSamplesUsed,
    [property: JsonPropertyName("from_prior")] bool FromPrior
);

public sealed record KBucketEntryDto(
    [property: JsonPropertyName("key")] KBucketKeyDto Key,
    [property: JsonPropertyName("value_kelvin_per_g2")] double ValueKelvinPerG2,
    [property: JsonPropertyName("stderr_kelvin_per_g2")] double StderrKelvinPerG2,
    [property: JsonPropertyName("n_samples")] int NSamples,
    [property: JsonPropertyName("from_prior")] bool FromPrior,
    [property: JsonPropertyName("from_single_track")] bool FromSingleTrack
);

public sealed record KBucketKeyDto(
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("corner")] string Corner,
    [property: JsonPropertyName("condition")] string Condition
);

public sealed record CTrackEntryDto(
    [property: JsonPropertyName("track_canonical")] string TrackCanonical,
    [property: JsonPropertyName("value")] double Value,
    [property: JsonPropertyName("stderr")] double Stderr,
    [property: JsonPropertyName("n_buckets_used")] int NBucketsUsed,
    [property: JsonPropertyName("anchor")] bool Anchor
);

public sealed record G2EntryDto(
    [property: JsonPropertyName("track_canonical")] string TrackCanonical,
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("condition")] string Condition,
    [property: JsonPropertyName("g2_typ")] double G2Typ,
    [property: JsonPropertyName("n_laps_used")] int NLapsUsed,
    [property: JsonPropertyName("g2_vs_lap_time")] G2CurveDto? G2VsLapTime = null
);

public sealed record LapTimeEntryDto(
    [property: JsonPropertyName("track_canonical")] string TrackCanonical,
    [property: JsonPropertyName("car")] string Car,
    [property: JsonPropertyName("condition")] string Condition,
    [property: JsonPropertyName("lap_time_typ_s")] double LapTimeTypS,
    [property: JsonPropertyName("n_laps_used")] int NLapsUsed
);

// Source-gen context — AOT-friendly. Single entry point lets us deserialize
// the full root DTO without runtime reflection.
[JsonSerializable(typeof(TireModelDto))]
public partial class TireModelJsonContext : JsonSerializerContext { }
