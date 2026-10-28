namespace InfernoAnalyzer.Core.Analysis;

/// <summary>
/// The numpy / pandas primitives the kernels are written against, with the
/// same edge semantics so parity fixtures match bit for bit where the
/// arithmetic order allows.
/// </summary>
public static class Numerics
{
    /// <summary>
    /// <c>pandas.Series.rolling(window, center=True, min_periods=1).mean()</c>:
    /// the window for index i spans <c>[i - (window - offset - 1), i + offset]</c>
    /// with <c>offset = (window - 1) / 2</c> (so an even window has one more
    /// sample before i than after), clipped to the array.
    /// </summary>
    public static double[] RollingMeanCentered(ReadOnlySpan<double> values, int window)
    {
        var n = values.Length;
        var result = new double[n];
        if (window <= 1)
        {
            values.CopyTo(result);
            return result;
        }

        var offset = (window - 1) / 2;
        for (var i = 0; i < n; i++)
        {
            var end = Math.Min(i + offset, n - 1);
            var start = Math.Max(i + offset + 1 - window, 0);
            var sum = 0.0;
            for (var j = start; j <= end; j++)
            {
                sum += values[j];
            }

            result[i] = sum / (end - start + 1);
        }

        return result;
    }

    /// <summary>
    /// <c>numpy.gradient(y)</c> with unit spacing: central differences inside,
    /// one-sided differences at the ends. A single sample gives zero.
    /// </summary>
    public static double[] Gradient(ReadOnlySpan<double> y)
    {
        var n = y.Length;
        var g = new double[n];
        if (n < 2)
        {
            return g;
        }

        g[0] = y[1] - y[0];
        g[n - 1] = y[n - 1] - y[n - 2];
        for (var i = 1; i < n - 1; i++)
        {
            g[i] = (y[i + 1] - y[i - 1]) / 2.0;
        }

        return g;
    }

    /// <summary><c>numpy.cumsum</c>.</summary>
    public static double[] CumulativeSum(ReadOnlySpan<double> values)
    {
        var result = new double[values.Length];
        var sum = 0.0;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
            result[i] = sum;
        }

        return result;
    }

    /// <summary><c>numpy.argmax</c>: index of the first maximum.</summary>
    public static int ArgMax(ReadOnlySpan<double> values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary><c>numpy.mean</c> of a contiguous array (pairwise sum / n).</summary>
    public static double Mean(ReadOnlySpan<double> values) =>
        values.Length == 0 ? double.NaN : Sum(values) / values.Length;

    /// <summary>
    /// <c>numpy.sum</c> of a contiguous 1-D array, reproducing numpy's pairwise
    /// summation (blocks of 128 with eight partial sums, halved recursively
    /// above that) so means and sums agree with Python to the last bit.
    /// </summary>
    public static double Sum(ReadOnlySpan<double> values)
    {
        const int blockSize = 128;
        var n = values.Length;
        if (n < 8)
        {
            var res = 0.0;
            for (var i = 0; i < n; i++)
            {
                res += values[i];
            }

            return res;
        }

        if (n <= blockSize)
        {
            Span<double> r = stackalloc double[8];
            for (var j = 0; j < 8; j++)
            {
                r[j] = values[j];
            }

            var i = 8;
            for (; i < n - (n % 8); i += 8)
            {
                for (var j = 0; j < 8; j++)
                {
                    r[j] += values[i + j];
                }
            }

            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++)
            {
                res += values[i];
            }

            return res;
        }

        var n2 = n / 2;
        n2 -= n2 % 8;
        return Sum(values[..n2]) + Sum(values[n2..]);
    }

    /// <summary><c>numpy.sign</c>.</summary>
    public static int Sign(double v) => v > 0 ? 1 : v < 0 ? -1 : 0;
}
