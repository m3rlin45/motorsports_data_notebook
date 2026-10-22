namespace InfernoAnalyzer.Core.Io;

/// <summary>One row of the laps table.</summary>
/// <param name="Num">Lap number as the logger counts (AIM: 0 is the out-lap).</param>
/// <param name="StartTime">Start, ms on the session clock.</param>
/// <param name="EndTime">End, ms on the session clock (exclusive).</param>
/// <param name="LapType">"full", "out", "in" (libibt also "partial", "incomplete"), or empty when unknown.</param>
/// <param name="Session">iRacing session index (libibt), null for AIM files.</param>
public sealed record Lap(int Num, long StartTime, long EndTime, string LapType, int? Session = null)
{
    /// <summary>Lap duration in milliseconds (the Python <c>lap_time</c> column).</summary>
    public long LapTimeMs => EndTime - StartTime;

    /// <summary>Lap duration in seconds.</summary>
    public double LapTimeS => LapTimeMs / 1000.0;
}
