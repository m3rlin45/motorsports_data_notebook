using System.Globalization;

namespace InfernoAnalyzer.Core.Io;

/// <summary>The storage type a channel's values had in the file; derived channels are <see cref="Float64"/>.</summary>
public enum ChannelValueType
{
    /// <summary>64-bit float (also every derived channel).</summary>
    Float64,

    /// <summary>32-bit float: interpolated values are rounded back to single precision, as Python's Arrow cast does.</summary>
    Float32,

    /// <summary>Any integer width.</summary>
    Integer,

    /// <summary>Boolean, stored as 0/1.</summary>
    Boolean,
}

/// <summary>
/// One logged channel: millisecond timecodes and values at the channel's own
/// sample rate. Mirrors a libxrk channel table (<c>timecodes</c> + value
/// column with field metadata); values are widened to double.
/// </summary>
public sealed class Channel
{
    /// <summary>Create a channel; <paramref name="metadata"/> is the Arrow field metadata (may be empty).</summary>
    public Channel(
        string name,
        long[] timecodes,
        double[] values,
        IReadOnlyDictionary<string, string>? metadata = null,
        ChannelValueType valueType = ChannelValueType.Float64)
    {
        if (timecodes.Length != values.Length)
        {
            throw new ArgumentException($"{name}: {timecodes.Length} timecodes but {values.Length} values");
        }

        Name = name;
        Timecodes = timecodes;
        Values = values;
        Metadata = metadata ?? new Dictionary<string, string>();
        ValueType = valueType;
    }

    /// <summary>How the values were stored in the file.</summary>
    public ChannelValueType ValueType { get; }

    /// <summary>Channel name as logged (display name; duplicates are suffixed by libxrk).</summary>
    public string Name { get; }

    /// <summary>Sample times, milliseconds on the session clock, non-decreasing.</summary>
    public long[] Timecodes { get; }

    /// <summary>Sample values in <see cref="Units"/>.</summary>
    public double[] Values { get; }

    /// <summary>Raw field metadata from the parser (units, dec_pts, interpolate, ...).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Number of samples.</summary>
    public int Length => Timecodes.Length;

    /// <summary>Unit string, empty when unknown.</summary>
    public string Units => Metadata.TryGetValue("units", out var u) ? u : string.Empty;

    /// <summary>Display decimal places.</summary>
    public int DecPts => Metadata.TryGetValue("dec_pts", out var d) && int.TryParse(d, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) ? n : 0;

    /// <summary>
    /// Whether resampling interpolates linearly (<c>interpolate == "True"</c>)
    /// rather than forward-filling. Derived channels carry no metadata and so
    /// forward-fill, as in Python.
    /// </summary>
    public bool Interpolate => Metadata.TryGetValue("interpolate", out var i) && i == "True";

    /// <summary>A channel with the same name, metadata and value type over new samples.</summary>
    public Channel With(long[] timecodes, double[] values) => new(Name, timecodes, values, Metadata, ValueType);

    /// <summary>Samples with <c>start &lt;= t &lt; end</c>.</summary>
    public Channel Slice(long start, long end)
    {
        var from = LowerBound(Timecodes, start);
        var to = LowerBound(Timecodes, end);
        return With(Timecodes[from..to], Values[from..to]);
    }

    /// <summary>
    /// The channel on <paramref name="target"/> timecodes: linear interpolation
    /// (numpy.interp semantics, edge values outside the range) when
    /// <see cref="Interpolate"/>, otherwise forward-fill with the first value
    /// before the first sample. Interpolated single-precision channels are
    /// rounded back to single precision, because Python casts the float64
    /// interpolation result back to the field's Arrow type.
    /// </summary>
    public Channel ResampleTo(long[] target)
    {
        var result = new double[target.Length];
        if (Length == 0)
        {
            Array.Fill(result, double.NaN);
            return With((long[])target.Clone(), result);
        }

        var xs = Timecodes;
        var ys = Values;
        for (var i = 0; i < target.Length; i++)
        {
            var t = target[i];
            if (Interpolate)
            {
                if (t <= xs[0])
                {
                    result[i] = ys[0];
                }
                else if (t >= xs[^1])
                {
                    result[i] = ys[^1];
                }
                else
                {
                    // Index of the first sample > t; the interval is [hi-1, hi].
                    // Same operation order as numpy.interp: slope * (x - xp[j]) + fp[j].
                    var hi = UpperBound(xs, t);
                    var lo = hi - 1;
                    var span = (double)(xs[hi] - xs[lo]);
                    var slope = span > 0 ? (ys[hi] - ys[lo]) / span : 0;
                    var value = slope * (t - xs[lo]) + ys[lo];
                    result[i] = ValueType == ChannelValueType.Float32 ? (float)value : value;
                }
            }
            else
            {
                // searchsorted(side="right") - 1, clamped; leading targets take the first value.
                var idx = UpperBound(xs, t) - 1;
                result[i] = ys[Math.Clamp(idx, 0, ys.Length - 1)];
            }
        }

        return With((long[])target.Clone(), result);
    }

    /// <summary>Index of the first element ≥ <paramref name="value"/> (numpy searchsorted left).</summary>
    public static int LowerBound(long[] sorted, long value)
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

    /// <summary>Index of the first element &gt; <paramref name="value"/> (numpy searchsorted right).</summary>
    public static int UpperBound(long[] sorted, long value)
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
}
