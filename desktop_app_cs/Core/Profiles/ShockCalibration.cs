namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Linear stroke-sensor calibration slopes for one axle (left and right share
/// them). Only slopes are stored, so zero references do not matter.
/// </summary>
/// <param name="WheelPerSensor">mm of wheel travel per mm of sensor travel (pot/wheel = 1 / this).</param>
/// <param name="DamperPerSensor">mm of damper travel per mm of sensor travel.</param>
/// <param name="DamperPerWheelFit">Directly fitted damper/wheel slope, if measured.</param>
/// <param name="WheelR2">R² of the wheel-vs-sensor fit.</param>
/// <param name="DamperR2">R² of the damper-vs-sensor fit.</param>
/// <param name="DamperWheelR2">R² of the direct damper-vs-wheel fit.</param>
/// <param name="Points">Raw rows [wheel_mm, damper_mm, sensor_mm].</param>
/// <param name="ExcludedPoints">0-based indices of rows left out of the fit.</param>
public sealed record AxleShockCalibration(
    double WheelPerSensor,
    double DamperPerSensor,
    double? DamperPerWheelFit = null,
    double? WheelR2 = null,
    double? DamperR2 = null,
    double? DamperWheelR2 = null,
    IReadOnlyList<IReadOnlyList<double>>? Points = null,
    IReadOnlyList<int>? ExcludedPoints = null)
{
    /// <summary>Pot travel per mm of wheel travel: the <see cref="MotionRatios"/> value.</summary>
    public double PotPerWheel => 1.0 / WheelPerSensor;

    /// <summary>
    /// Damper travel per mm of wheel travel: the directly fitted value when
    /// available, else the ratio of the two sensor slopes.
    /// </summary>
    public double DamperPerWheel => DamperPerWheelFit ?? DamperPerSensor / WheelPerSensor;
}

/// <summary>Front and rear <see cref="AxleShockCalibration"/> for one vehicle.</summary>
public sealed record ShockCalibration(AxleShockCalibration Front, AxleShockCalibration Rear)
{
    /// <summary>Pot/wheel ratios for the suspension analysis.</summary>
    public MotionRatios PotWheelRatios() =>
        new(Front.PotPerWheel, Front.PotPerWheel, Rear.PotPerWheel, Rear.PotPerWheel);
}
