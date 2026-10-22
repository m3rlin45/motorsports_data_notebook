namespace InfernoAnalyzer.Core.Profiles;

/// <summary>
/// Pot/wheel motion ratios per corner, as applied to the logged shock pot
/// channel (<c>wheel = pot / ratio</c>). The only ratio the suspension
/// analysis uses.
/// </summary>
public sealed record MotionRatios(
    double FrontLeft,
    double FrontRight,
    double RearLeft,
    double RearRight)
{
    /// <summary>The Toyota 86 ZN6 defaults (0.997 front MacPherson, 0.768 rear multi-link).</summary>
    public static MotionRatios Toyota86Zn6 { get; } = new(0.997, 0.997, 0.768, 0.768);
}

/// <summary>Front and rear track widths in mm, hub centre to hub centre (roll gradient).</summary>
public sealed record TrackWidths
{
    /// <summary>Create validated track widths.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A width is not positive.</exception>
    public TrackWidths(double frontMm, double rearMm)
    {
        if (!(frontMm > 0) || !(rearMm > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(frontMm), "Track widths must be positive (mm)");
        }

        FrontMm = frontMm;
        RearMm = rearMm;
    }

    /// <summary>Front track width, mm.</summary>
    public double FrontMm { get; }

    /// <summary>Rear track width, mm.</summary>
    public double RearMm { get; }
}
