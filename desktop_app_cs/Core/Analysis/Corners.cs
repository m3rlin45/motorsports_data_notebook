using InfernoAnalyzer.Core.Io;

namespace InfernoAnalyzer.Core.Analysis;

/// <summary>A detected corner; port of the Python <c>Corner</c> dataclass.</summary>
public sealed class Corner
{
    /// <summary>Corner number (1-indexed).</summary>
    public int Id { get; set; }

    /// <summary>Corner name, e.g. "Turn 1".</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>'L' or 'R'.</summary>
    public char Direction { get; set; }

    /// <summary>Array index where the corner begins.</summary>
    public int StartIdx { get; set; }

    /// <summary>Array index where the corner ends.</summary>
    public int EndIdx { get; set; }

    /// <summary>Distance along the track where the corner begins, metres.</summary>
    public double StartDist { get; set; }

    /// <summary>Distance along the track where the corner ends, metres.</summary>
    public double EndDist { get; set; }

    /// <summary>Array index of the apex (maximum curvature).</summary>
    public int ApexIdx { get; set; }

    /// <summary>Distance along the track of the apex, metres.</summary>
    public double ApexDist { get; set; }

    /// <summary>Maximum curvature at the apex, 1/m.</summary>
    public double MaxCurvature { get; set; }

    /// <summary>Corner length in metres.</summary>
    public double Length => EndDist - StartDist;

    /// <summary>Approximate radius at the apex, metres (10 km for a straight).</summary>
    public double Radius => MaxCurvature > 1e-6 ? 1.0 / MaxCurvature : 10000.0;

    /// <summary>A copy.</summary>
    public Corner Clone() => (Corner)MemberwiseClone();
}

/// <summary>Curvature along a lap, from <see cref="Corners.ComputeCurvature"/>.</summary>
/// <param name="Curvature">Smoothed absolute curvature, 1/m.</param>
/// <param name="SignedCurvature">Signed curvature: positive left, negative right.</param>
/// <param name="Radius">Radius of curvature, metres, capped at 10 km.</param>
public sealed record CurvatureResult(double[] Curvature, double[] SignedCurvature, double[] Radius);

/// <summary>
/// Corner detection from GPS via curvature analysis; port of <c>corners.py</c>.
/// </summary>
public static class Corners
{
    /// <summary>Default curvature threshold, 1/m (≈ 167 m radius).</summary>
    public const double DefaultThreshold = 0.006;

    /// <summary>Default minimum corner length, metres.</summary>
    public const double DefaultMinCornerLength = 15;

    /// <summary>Default minimum gap between same-direction corners, metres.</summary>
    public const double DefaultMinGap = 80;

    /// <summary>Default position smoothing window, samples.</summary>
    public const int DefaultPosSmoothWindow = 15;

    /// <summary>Default curvature smoothing window, samples.</summary>
    public const int DefaultCurvSmoothWindow = 30;

    /// <summary>Earth radius used by the projection, metres.</summary>
    private const double EarthRadius = 6371000;

    /// <summary><c>numpy.radians</c> multiplies by this constant (not divide-by-180 after multiplying by π).</summary>
    private const double DegToRad = Math.PI / 180.0;

    /// <summary>
    /// <c>compute_lap_distance</c>: cumulative distance from integrating speed
    /// (m/s) over the sample intervals, first interval zero.
    /// </summary>
    public static double[] ComputeLapDistance(ReadOnlySpan<long> timecodes, ReadOnlySpan<double> speed)
    {
        var n = timecodes.Length;
        var distance = new double[n];
        if (n == 0)
        {
            return distance;
        }

        var sum = 0.0;
        for (var i = 1; i < n; i++)
        {
            var dt = ((timecodes[i] - timecodes[0]) / 1000.0) - ((timecodes[i - 1] - timecodes[0]) / 1000.0);
            sum += speed[i] * dt;
            distance[i] = sum;
        }

        return distance;
    }

    /// <summary>
    /// <c>gps_to_local_xy</c>: equirectangular projection centred on the mean
    /// latitude/longitude, metres.
    /// </summary>
    public static (double[] X, double[] Y) GpsToLocalXy(ReadOnlySpan<double> lat, ReadOnlySpan<double> lon)
    {
        var lat0 = Numerics.Mean(lat) * DegToRad;
        var lon0 = Numerics.Mean(lon) * DegToRad;
        var cosLat0 = Math.Cos(lat0);
        var x = new double[lat.Length];
        var y = new double[lat.Length];
        for (var i = 0; i < lat.Length; i++)
        {
            var latRad = lat[i] * DegToRad;
            var lonRad = lon[i] * DegToRad;
            x[i] = EarthRadius * (lonRad - lon0) * cosLat0;
            y[i] = EarthRadius * (latRad - lat0);
        }

        return (x, y);
    }

    /// <summary>
    /// <c>compute_curvature</c>: κ = (x'y'' − y'x'') / (x'² + y'²)^1.5 on
    /// smoothed positions, then smoothed again.
    /// </summary>
    public static CurvatureResult ComputeCurvature(
        ReadOnlySpan<double> x,
        ReadOnlySpan<double> y,
        int posSmoothWindow = DefaultPosSmoothWindow,
        int curvSmoothWindow = DefaultCurvSmoothWindow)
    {
        var xs = posSmoothWindow > 1 ? Numerics.RollingMeanCentered(x, posSmoothWindow) : x.ToArray();
        var ys = posSmoothWindow > 1 ? Numerics.RollingMeanCentered(y, posSmoothWindow) : y.ToArray();

        var dx = Numerics.Gradient(xs);
        var dy = Numerics.Gradient(ys);
        var ddx = Numerics.Gradient(dx);
        var ddy = Numerics.Gradient(dy);

        var n = x.Length;
        var signed = new double[n];
        for (var i = 0; i < n; i++)
        {
            var numerator = dx[i] * ddy[i] - dy[i] * ddx[i];
            var denominator = Math.Pow(dx[i] * dx[i] + dy[i] * dy[i], 1.5);
            signed[i] = denominator > 1e-10 ? numerator / denominator : 0;
        }

        if (curvSmoothWindow > 1)
        {
            signed = Numerics.RollingMeanCentered(signed, curvSmoothWindow);
        }

        var curvature = new double[n];
        var radius = new double[n];
        for (var i = 0; i < n; i++)
        {
            curvature[i] = Math.Abs(signed[i]);
            var r = curvature[i] > 1e-4 ? 1 / curvature[i] : 10000;
            radius[i] = Math.Clamp(r, 0, 10000);
        }

        return new CurvatureResult(curvature, signed, radius);
    }

    /// <summary>
    /// <c>identify_corners</c>: GPS → local XY → distance along the path →
    /// curvature → corners. Samples at exactly (0, 0) are dropped first
    /// (iRacing reports them before the car is on track).
    /// </summary>
    public static List<Corner> IdentifyCorners(
        ReadOnlySpan<double> lat,
        ReadOnlySpan<double> lon,
        double threshold = DefaultThreshold,
        double minCornerLength = DefaultMinCornerLength,
        double minGap = DefaultMinGap,
        int posSmoothWindow = DefaultPosSmoothWindow,
        int curvSmoothWindow = DefaultCurvSmoothWindow)
    {
        var (validLat, validLon) = DropInvalidGps(lat, lon);
        var (x, y) = GpsToLocalXy(validLat, validLon);
        var distance = PathDistance(x, y);
        var curvature = ComputeCurvature(x, y, posSmoothWindow, curvSmoothWindow);
        return IdentifyCornersFromCurvature(distance, curvature.Curvature, curvature.SignedCurvature, threshold, minCornerLength, minGap);
    }

    /// <summary>Corners of a lap from its GPS channels (names from the profile).</summary>
    public static List<Corner> IdentifyCorners(LogFile lap, string latitudeChannel, string longitudeChannel) =>
        IdentifyCorners(lap[latitudeChannel].Values, lap[longitudeChannel].Values);

    /// <summary>Cumulative distance along the XY path (first step zero).</summary>
    public static double[] PathDistance(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        var steps = new double[x.Length];
        for (var i = 1; i < x.Length; i++)
        {
            var dx = x[i] - x[i - 1];
            var dy = y[i] - y[i - 1];
            steps[i] = Math.Sqrt(dx * dx + dy * dy);
        }

        return Numerics.CumulativeSum(steps);
    }

    /// <summary>
    /// <c>identify_corners_from_curvature</c>: runs of curvature above the
    /// threshold become corners, split where the sign flips; short ones are
    /// dropped and same-direction neighbours closer than <paramref name="minGap"/>
    /// are merged; then numbered.
    /// </summary>
    public static List<Corner> IdentifyCornersFromCurvature(
        ReadOnlySpan<double> distance,
        ReadOnlySpan<double> curvature,
        ReadOnlySpan<double> signedCurvature,
        double threshold = DefaultThreshold,
        double minCornerLength = DefaultMinCornerLength,
        double minGap = DefaultMinGap)
    {
        var corners = new List<Corner>();
        int? cornerStart = null;
        var cornerDirection = 0;

        for (var i = 0; i < curvature.Length; i++)
        {
            if (curvature[i] > threshold)
            {
                var dir = Numerics.Sign(signedCurvature[i]);
                if (cornerStart is null)
                {
                    cornerStart = i;
                    cornerDirection = dir;
                }
                else if (dir != cornerDirection && dir != 0)
                {
                    AddIfLongEnough(corners, distance, curvature, cornerStart.Value, i - 1, cornerDirection, minCornerLength);
                    cornerStart = i;
                    cornerDirection = dir;
                }
            }
            else if (cornerStart is not null)
            {
                AddIfLongEnough(corners, distance, curvature, cornerStart.Value, i - 1, cornerDirection, minCornerLength);
                cornerStart = null;
                cornerDirection = 0;
            }
        }

        if (cornerStart is not null)
        {
            AddIfLongEnough(corners, distance, curvature, cornerStart.Value, distance.Length - 1, cornerDirection, minCornerLength);
        }

        var merged = new List<Corner>();
        foreach (var corner in corners)
        {
            if (merged.Count > 0
                && corner.StartDist - merged[^1].EndDist < minGap
                && corner.Direction == merged[^1].Direction)
            {
                var prev = merged[^1];
                prev.EndDist = corner.EndDist;
                prev.EndIdx = corner.EndIdx;
                if (corner.MaxCurvature > prev.MaxCurvature)
                {
                    prev.ApexDist = corner.ApexDist;
                    prev.ApexIdx = corner.ApexIdx;
                    prev.MaxCurvature = corner.MaxCurvature;
                }
            }
            else
            {
                merged.Add(corner.Clone());
            }
        }

        for (var i = 0; i < merged.Count; i++)
        {
            merged[i].Id = i + 1;
            merged[i].Name = $"Turn {i + 1}";
        }

        return merged;
    }

    /// <summary>Append the corner [start, end] when it is at least <paramref name="minCornerLength"/> long.</summary>
    private static void AddIfLongEnough(
        List<Corner> corners,
        ReadOnlySpan<double> distance,
        ReadOnlySpan<double> curvature,
        int start,
        int end,
        int direction,
        double minCornerLength)
    {
        if (distance[end] - distance[start] < minCornerLength)
        {
            return;
        }

        var apex = start + Numerics.ArgMax(curvature[start..(end + 1)]);
        corners.Add(new Corner
        {
            Direction = direction > 0 ? 'L' : 'R',
            StartIdx = start,
            EndIdx = end,
            StartDist = distance[start],
            EndDist = distance[end],
            ApexIdx = apex,
            ApexDist = distance[apex],
            MaxCurvature = curvature[apex],
        });
    }

    private static (double[] Lat, double[] Lon) DropInvalidGps(ReadOnlySpan<double> lat, ReadOnlySpan<double> lon)
    {
        var keptLat = new List<double>(lat.Length);
        var keptLon = new List<double>(lon.Length);
        for (var i = 0; i < lat.Length; i++)
        {
            if (lat[i] != 0.0 || lon[i] != 0.0)
            {
                keptLat.Add(lat[i]);
                keptLon.Add(lon[i]);
            }
        }

        return (keptLat.ToArray(), keptLon.ToArray());
    }
}
