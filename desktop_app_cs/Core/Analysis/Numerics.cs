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

    /// <summary>
    /// <c>numpy.gradient(f, x)</c> with a coordinate array: numpy's
    /// second-order interior formula (used even when the spacing happens to
    /// be uniform) and first-order one-sided ends.
    /// </summary>
    public static double[] Gradient(ReadOnlySpan<double> f, ReadOnlySpan<double> x)
    {
        var n = f.Length;
        var g = new double[n];
        if (n < 2)
        {
            return g;
        }

        g[0] = (f[1] - f[0]) / (x[1] - x[0]);
        g[n - 1] = (f[n - 1] - f[n - 2]) / (x[n - 1] - x[n - 2]);
        for (var i = 1; i < n - 1; i++)
        {
            var dx1 = x[i] - x[i - 1];
            var dx2 = x[i + 1] - x[i];
            var a = -dx2 / (dx1 * (dx1 + dx2));
            var b = (dx2 - dx1) / (dx1 * dx2);
            var c = dx1 / (dx2 * (dx1 + dx2));
            g[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
        }

        return g;
    }

    /// <summary><c>numpy.std</c> (population, ddof = 0) with numpy's pairwise sums.</summary>
    public static double Std(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
        {
            return double.NaN;
        }

        var mean = Mean(values);
        var squares = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var d = values[i] - mean;
            squares[i] = d * d;
        }

        return Math.Sqrt(Sum(squares) / values.Length);
    }

    /// <summary><c>numpy.arange(start, stop, step)</c> for floats: <c>start + k * step</c>, k &lt; ceil((stop − start) / step).</summary>
    public static double[] ARange(double start, double stop, double step)
    {
        var count = (int)Math.Ceiling((stop - start) / step);
        var result = new double[Math.Max(count, 0)];
        for (var k = 0; k < result.Length; k++)
        {
            result[k] = start + k * step;
        }

        return result;
    }

    /// <summary>
    /// <c>numpy.histogram(values, bins=edges)</c> counts: bin i holds
    /// <c>edges[i] &lt;= v &lt; edges[i+1]</c>, the last bin also its right
    /// edge; values outside the edges are ignored.
    /// </summary>
    public static long[] Histogram(ReadOnlySpan<double> values, ReadOnlySpan<double> edges)
    {
        var counts = new long[Math.Max(edges.Length - 1, 0)];
        if (counts.Length == 0)
        {
            return counts;
        }

        var last = edges[^1];
        foreach (var v in values)
        {
            if (v < edges[0] || v > last || double.IsNaN(v))
            {
                continue;
            }

            var bin = v == last ? counts.Length - 1 : SearchSortedRight(edges, v) - 1;
            counts[bin]++;
        }

        return counts;
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

    /// <summary><c>numpy.argmin</c>: index of the first minimum.</summary>
    public static int ArgMin(ReadOnlySpan<double> values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] < values[best])
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary><c>numpy.max</c> (NaN for an empty span).</summary>
    public static double Max(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
        {
            return double.NaN;
        }

        var best = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            best = Math.Max(best, values[i]);
        }

        return best;
    }

    /// <summary><c>numpy.min</c> (NaN for an empty span).</summary>
    public static double Min(ReadOnlySpan<double> values)
    {
        if (values.Length == 0)
        {
            return double.NaN;
        }

        var best = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            best = Math.Min(best, values[i]);
        }

        return best;
    }

    /// <summary><c>numpy.searchsorted(a, v)</c> (side="left"): first index with a[i] ≥ v.</summary>
    public static int SearchSortedLeft(ReadOnlySpan<double> sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >> 1;
            if (sorted[mid] < value)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary><c>numpy.searchsorted(a, v, side="right")</c>: first index with a[i] &gt; v.</summary>
    public static int SearchSortedRight(ReadOnlySpan<double> sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            var mid = (lo + hi) >> 1;
            if (sorted[mid] <= value)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary>
    /// <c>numpy.convolve(a, kernel, mode="same")</c>: the central
    /// <c>len(a)</c> samples of the full convolution, zero-padded.
    /// </summary>
    public static double[] ConvolveSame(ReadOnlySpan<double> a, ReadOnlySpan<double> kernel)
    {
        var n = a.Length;
        var m = kernel.Length;
        var result = new double[n];
        if (n == 0 || m == 0)
        {
            return result;
        }

        // full[k] = Σ a[i] kernel[k - i]; "same" takes full[(m-1)/2 .. (m-1)/2 + n).
        var offset = (m - 1) / 2;
        for (var i = 0; i < n; i++)
        {
            var k = i + offset;
            var sum = 0.0;
            var jLo = Math.Max(0, k - n + 1);
            var jHi = Math.Min(m - 1, k);
            for (var j = jLo; j <= jHi; j++)
            {
                sum += a[k - j] * kernel[j];
            }

            result[i] = sum;
        }

        return result;
    }

    /// <summary>
    /// <c>infer_channel_scale</c>: 1.0 for 0–1 data, 100.0 for 0–100 data
    /// (max |x| above 1.5).
    /// </summary>
    public static double InferChannelScale(ReadOnlySpan<double> values)
    {
        var maxAbs = 0.0;
        foreach (var v in values)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(v));
        }

        return maxAbs <= 1.5 ? 1.0 : 100.0;
    }
}
