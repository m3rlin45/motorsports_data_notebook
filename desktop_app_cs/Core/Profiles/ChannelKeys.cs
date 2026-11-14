namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// The canonical channel keys and their default AIM names (Toyota 86 ZN6 /
/// AiM MXP). Mirrors <c>DEFAULT_CHANNEL_NAMES</c> in the Python package's
/// <c>profiles.py</c>; keep the two in sync.
/// </summary>
public static class ChannelKeys
{
    /// <summary>Canonical key → default channel name, in canonical order.</summary>
    public static readonly IReadOnlyDictionary<string, string> DefaultChannelNames = BuildDefaults();

    /// <summary>All canonical keys in definition order.</summary>
    public static readonly IReadOnlyList<string> All = DefaultChannelNames.Keys.ToList();

    private static Dictionary<string, string> BuildDefaults()
    {
        var names = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["throttle"] = "PPS",

            // Throttle (butterfly) angle in degrees from the ECU, where the car logs it
            // next to, or instead of, a percentage (the KK-F's TA).
            ["throttle_angle"] = "",
            ["brake"] = "BrakePress",
            ["brake_rear"] = "",
            ["gps_speed"] = "GPS Speed",
            ["gps_latitude"] = "GPS Latitude",
            ["gps_longitude"] = "GPS Longitude",
            ["lateral_g"] = "LateralAcc",
            ["inline_g"] = "InlineAcc",

            // Raw inertial channels for the tire ETL; lateral_g / inline_g above are
            // the analysis choice per vehicle and may be the GPS-derived ones.
            ["imu_lateral_g"] = "LateralAcc",
            ["imu_inline_g"] = "InlineAcc",
            ["vertical_g"] = "VerticalAcc",
            ["yaw_rate"] = "YawRate",
            ["pitch_rate"] = "PitchRate",
            ["roll_rate"] = "RollRate",
            ["gps_altitude"] = "GPS Altitude",
            ["gps_lateral_g"] = "GPS_LateralAcc",
            ["gps_inline_g"] = "GPS_InlineAcc",
            ["gps_yaw_rate"] = "GPS_Yaw_Rate",
            ["steering"] = "SteerAngle",
            ["rpm"] = "RPM",
            ["gear"] = "Gear",
            ["wheel_speed_fl"] = "WheelSpdFL",
            ["wheel_speed_fr"] = "WheelSpdFR",
            ["wheel_speed_rl"] = "WheelSpdRL",
            ["wheel_speed_rr"] = "WheelSpdRR",
            ["shock_fl"] = "LF_Shock_Pot",
            ["shock_fr"] = "RF_Shock_Pot",
            ["shock_rl"] = "LR_Shock_Pot",
            ["shock_rr"] = "RR_Shock_Pot",
            ["tpms_press_fl"] = "TPMS_Press_LF",
            ["tpms_press_fr"] = "TPMS_Press_RF",
            ["tpms_press_rl"] = "TPMS_Press_LR",
            ["tpms_press_rr"] = "TPMS_Press_RR",
            ["tpms_temp_fl"] = "TPMS_Temp_LF",
            ["tpms_temp_fr"] = "TPMS_Temp_RF",
            ["tpms_temp_rl"] = "TPMS_Temp_LR",
            ["tpms_temp_rr"] = "TPMS_Temp_RR",
        };

        foreach (var corner in new[] { ("fl", "FL"), ("fr", "FR"), ("rl", "RL"), ("rr", "RR") })
        {
            for (var i = 1; i <= 8; i++)
            {
                names[$"tire_temp_{corner.Item1}_{i}"] = $"{corner.Item2}_Ch{i}";
            }
        }

        return names;
    }
}
