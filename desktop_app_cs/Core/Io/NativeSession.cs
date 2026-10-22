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

    /// <summary>Export channel <paramref name="index"/> into <paramref name="stream"/>; 0 on success.</summary>
    unsafe int ExportChannel(int index, CArrowArrayStream* stream);

    /// <summary>Export the laps table into <paramref name="stream"/>; 0 on success.</summary>
    unsafe int ExportLaps(CArrowArrayStream* stream);

    /// <summary>The metadata JSON document.</summary>
    string MetadataJson { get; }
}

/// <summary>Turns a native session into a <see cref="LogFile"/>, one Arrow batch at a time.</summary>
internal static class NativeSessionReader
{
    /// <summary>Read every channel, the laps and the metadata, then close the session.</summary>
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
                channels.Add(ArrowConvert.ToChannel(batch, name));
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
            if (rc != 0)
            {
                throw new InvalidDataException($"{parserName} could not export {what} (code {rc})");
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
