using Apache.Arrow;
using Apache.Arrow.C;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// The shape both parser FFI libraries expose: an opaque session handle,
/// channels and laps as Arrow array streams, metadata as JSON.
/// </summary>
internal interface INativeSession : IDisposable
{
    /// <summary>Number of channels.</summary>
    int ChannelCount { get; }

    /// <summary>Name of channel <paramref name="index"/>.</summary>
    string ChannelName(int index);

    /// <summary>Export channel <paramref name="index"/> into <paramref name="stream"/>; a <see cref="NativeStatus"/> code.</summary>
    unsafe int ExportChannel(int index, CArrowArrayStream* stream);

    /// <summary>Export the laps table into <paramref name="stream"/>; a <see cref="NativeStatus"/> code.</summary>
    unsafe int ExportLaps(CArrowArrayStream* stream);

    /// <summary>The metadata JSON document.</summary>
    string MetadataJson { get; }
}

/// <summary>
/// The status codes of the stream exports. Both libraries define the same
/// values (<c>XRK_OK</c> … <c>XRK_ERR_PANIC</c>, <c>IBT_OK</c> … <c>IBT_ERR_PANIC</c>).
/// </summary>
internal static class NativeStatus
{
    /// <summary>The stream was exported.</summary>
    public const int Ok = 0;

    /// <summary>The channel index is not below the channel count.</summary>
    public const int Index = 1;

    /// <summary>The session or the output pointer was NULL.</summary>
    public const int Null = 2;

    /// <summary>The library panicked; the panic was caught at the boundary.</summary>
    public const int Panic = 3;

    /// <summary>What <paramref name="code"/> means, for an error message.</summary>
    public static string Describe(int code) => code switch
    {
        Ok => "no error",
        Index => "channel index out of range",
        Null => "null session or stream pointer",
        Panic => "the parser panicked",
        _ => $"unknown status {code}",
    };
}

/// <summary>Turns a native session into a <see cref="LogFile"/>, one Arrow batch at a time.</summary>
internal static class NativeSessionReader
{
    /// <summary>
    /// Read every channel, the laps and the metadata, then close the session.
    /// Vector-valued channels are left out: libibt exports an array variable
    /// that is not a time subsample (the per-car <c>CarIdx*</c> arrays) as one
    /// FixedSizeList per tick, and a <see cref="Channel"/> is one value per
    /// sample. Time-subsample arrays (<c>SteeringWheelTorque_ST</c>, 360 Hz)
    /// arrive as ordinary channels at their own rate and are kept.
    /// </summary>
    public static unsafe LogFile Read(INativeSession session, string fileName, string parserName)
    {
        using (session)
        {
            var count = session.ChannelCount;
            var channels = new List<Channel>(count);
            for (var i = 0; i < count; i++)
            {
                var name = session.ChannelName(i);
                var index = i;
                using var batch = ReadBatch(stream => session.ExportChannel(index, stream), name, parserName);
                if (ArrowConvert.ToChannel(batch, name) is { } channel)
                {
                    channels.Add(channel);
                }
            }

            using var lapsBatch = ReadBatch(session.ExportLaps, "laps", parserName);
            var laps = ArrowConvert.ToLaps(lapsBatch);
            var json = session.MetadataJson;
            return new LogFile(channels, laps, ArrowConvert.FlattenMetadata(json), fileName, json);
        }
    }

    private static unsafe RecordBatch ReadBatch(ExportStream export, string what, string parserName)
    {
        var stream = CArrowArrayStream.Create();
        try
        {
            var rc = export(stream);
            if (rc != NativeStatus.Ok)
            {
                throw new InvalidDataException($"{parserName} could not export {what}: {NativeStatus.Describe(rc)}");
            }

            using var reader = CArrowArrayStreamImporter.ImportArrayStream(stream);
            return reader.ReadNextRecordBatchAsync().GetAwaiter().GetResult()
                ?? throw new InvalidDataException($"{parserName} exported no data for {what}");
        }
        finally
        {
            CArrowArrayStream.Free(stream);
        }
    }

    private unsafe delegate int ExportStream(CArrowArrayStream* stream);
}
