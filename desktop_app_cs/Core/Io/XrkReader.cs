using System.Globalization;
using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.C;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// Reads AIM .xrk/.xrz files through <c>libxrk-ffi</c>: every channel and
/// the laps table cross the boundary as Arrow record batches via the C Data
/// Interface, the metadata as JSON.
/// </summary>
public static class XrkReader
{
    /// <summary>Parse the file at <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">libxrk rejected the file.</exception>
    public static LogFile Read(string path)
    {
        var handle = XrkNative.xrk_open_path(path, out var error);
        return Build(handle, error, path);
    }

    /// <summary>Parse in-memory file bytes.</summary>
    /// <exception cref="InvalidDataException">libxrk rejected the data.</exception>
    public static unsafe LogFile Read(ReadOnlySpan<byte> data, string fileName = "<bytes>")
    {
        nint handle;
        nint error;
        fixed (byte* p = data)
        {
            handle = XrkNative.xrk_open_bytes(p, (nuint)data.Length, out error);
        }

        return Build(handle, error, fileName);
    }

    /// <summary>The libxrk-ffi crate version, as a smoke test of the native library.</summary>
    public static string NativeVersion => XrkNative.Version;

    private static LogFile Build(nint handle, nint error, string fileName)
    {
        if (handle == nint.Zero)
        {
            var message = XrkNative.TakeString(error);
            throw new InvalidDataException(message.Length > 0 ? message : $"libxrk could not read {fileName}");
        }

        return NativeSessionReader.Read(new Session(handle), fileName, "libxrk");
    }

    private sealed unsafe class Session(nint handle) : INativeSession
    {
        public int ChannelCount => (int)XrkNative.xrk_channel_count(handle);
        public string ChannelName(int index) => XrkNative.TakeString(XrkNative.xrk_channel_name(handle, (nuint)index));
        public int ExportChannel(int index, CArrowArrayStream* stream) => XrkNative.xrk_channel_stream(handle, (nuint)index, stream);
        public int ExportLaps(CArrowArrayStream* stream) => XrkNative.xrk_laps_stream(handle, stream);
        public string MetadataJson => XrkNative.TakeString(XrkNative.xrk_metadata_json(handle));
        public void Dispose() => XrkNative.xrk_close(handle);
    }
}

/// <summary>Arrow record batches → the session model.</summary>
internal static class ArrowConvert
{
    /// <summary>A channel batch (<c>timecodes</c> + value column named <paramref name="name"/>).</summary>
    public static Channel ToChannel(RecordBatch batch, string name)
    {
        var timecodes = ToInt64(batch.Column("timecodes"));
        var valueIndex = batch.Schema.GetFieldIndex(name, StringComparer.Ordinal);
        if (valueIndex < 0)
        {
            valueIndex = batch.Schema.GetFieldIndex("timecodes", StringComparer.Ordinal) == 0 ? 1 : 0;
        }

        var field = batch.Schema.GetFieldByIndex(valueIndex);
        var column = batch.Column(valueIndex);
        var values = ToDouble(column);
        var metadata = field.HasMetadata
            ? new Dictionary<string, string>(field.Metadata, StringComparer.Ordinal)
            : new Dictionary<string, string>(StringComparer.Ordinal);
        var valueType = column switch
        {
            DoubleArray => ChannelValueType.Float64,
            FloatArray => ChannelValueType.Float32,
            BooleanArray => ChannelValueType.Boolean,
            _ => ChannelValueType.Integer,
        };
        return new Channel(name, timecodes, values, metadata, valueType);
    }

    /// <summary>The laps batch (<c>num</c>, <c>start_time</c>, <c>end_time</c>, <c>lap_type</c>, optional <c>session</c>).</summary>
    public static List<Lap> ToLaps(RecordBatch batch)
    {
        var nums = ToInt64(batch.Column("num"));
        var starts = ToInt64(batch.Column("start_time"));
        var ends = ToInt64(batch.Column("end_time"));
        var hasType = batch.Schema.GetFieldIndex("lap_type", StringComparer.Ordinal) >= 0;
        var types = hasType ? (StringArray)batch.Column("lap_type") : null;
        var hasSession = batch.Schema.GetFieldIndex("session", StringComparer.Ordinal) >= 0;
        var sessions = hasSession ? ToInt64(batch.Column("session")) : null;

        var laps = new List<Lap>(batch.Length);
        for (var i = 0; i < batch.Length; i++)
        {
            laps.Add(new Lap(
                (int)nums[i],
                starts[i],
                ends[i],
                types?.GetString(i) ?? string.Empty,
                sessions is null ? null : (int)sessions[i]));
        }

        return laps;
    }

    /// <summary>
    /// Metadata JSON → text values: strings as-is, numbers and booleans in
    /// their JSON form (what Python's <c>str()</c> gives for ints), null as
    /// empty, arrays/objects as raw JSON.
    /// </summary>
    public static Dictionary<string, string> FlattenMetadata(string json)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        using var document = JsonDocument.Parse(json);
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            return result;
        }

        foreach (var property in document.RootElement.EnumerateObject())
        {
            result[property.Name] = property.Value.ValueKind switch
            {
                JsonValueKind.String => property.Value.GetString() ?? string.Empty,
                JsonValueKind.Null => string.Empty,
                JsonValueKind.True => "True",
                JsonValueKind.False => "False",
                _ => property.Value.GetRawText(),
            };
        }

        return result;
    }

    private static long[] ToInt64(IArrowArray array) => array switch
    {
        Int64Array a => a.Values.ToArray(),
        Int32Array a => Widen(a.Values),
        Int16Array a => Widen(a.Values),
        UInt32Array a => Widen(a.Values),
        UInt16Array a => Widen(a.Values),
        UInt8Array a => Widen(a.Values),
        _ => throw new NotSupportedException($"integer column of type {array.Data.DataType.Name}"),
    };

    private static double[] ToDouble(IArrowArray array) => array switch
    {
        DoubleArray a => a.Values.ToArray(),
        FloatArray a => WidenD(a.Values),
        Int64Array a => WidenD(a.Values),
        Int32Array a => WidenD(a.Values),
        Int16Array a => WidenD(a.Values),
        UInt32Array a => WidenD(a.Values),
        UInt16Array a => WidenD(a.Values),
        UInt8Array a => WidenD(a.Values),
        BooleanArray a => Enumerable.Range(0, a.Length).Select(i => a.GetValue(i) == true ? 1.0 : 0.0).ToArray(),
        _ => throw new NotSupportedException($"value column of type {array.Data.DataType.Name}"),
    };

    private static long[] Widen<T>(ReadOnlySpan<T> values) where T : struct, IConvertible
    {
        var result = new long[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = values[i].ToInt64(CultureInfo.InvariantCulture);
        }

        return result;
    }

    private static double[] WidenD<T>(ReadOnlySpan<T> values) where T : struct, IConvertible
    {
        var result = new double[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            result[i] = values[i].ToDouble(CultureInfo.InvariantCulture);
        }

        return result;
    }
}
