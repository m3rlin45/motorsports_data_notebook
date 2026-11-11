using Apache.Arrow.C;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// Reads iRacing .ibt files through <c>libibt-ffi</c>, the same way
/// <see cref="XrkReader"/> reads AIM files.
/// </summary>
public static class IbtReader
{
    /// <summary>Parse the file at <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">libibt rejected the file.</exception>
    public static LogFile Read(string path)
    {
        var handle = IbtNative.ibt_open_path(path, out var error);
        return Build(handle, error, path);
    }

    /// <summary>Parse in-memory file bytes.</summary>
    /// <exception cref="InvalidDataException">libibt rejected the data.</exception>
    public static unsafe LogFile Read(ReadOnlySpan<byte> data, string fileName = "<bytes>")
    {
        nint handle;
        nint error;
        fixed (byte* p = data)
        {
            handle = IbtNative.ibt_open_bytes(p, (nuint)data.Length, out error);
        }

        return Build(handle, error, fileName);
    }

    /// <summary>The libibt-ffi crate version, as a smoke test of the native library.</summary>
    public static string NativeVersion => IbtNative.Version;

    private static LogFile Build(nint handle, nint error, string fileName)
    {
        if (handle == nint.Zero)
        {
            var message = IbtNative.TakeString(error);
            throw new InvalidDataException(message.Length > 0 ? message : $"libibt could not read {fileName}");
        }

        return NativeSessionReader.Read(new Session(handle), fileName, "libibt");
    }

    private sealed unsafe class Session(nint handle) : INativeSession
    {
        public int ChannelCount => (int)IbtNative.ibt_channel_count(handle);
        public string ChannelName(int index) => IbtNative.TakeString(IbtNative.ibt_channel_name(handle, (nuint)index));
        public int ExportChannel(int index, CArrowArrayStream* stream) => IbtNative.ibt_channel_stream(handle, (nuint)index, stream);
        public int ExportLaps(CArrowArrayStream* stream) => IbtNative.ibt_laps_stream(handle, stream);
        public string MetadataJson => IbtNative.TakeString(IbtNative.ibt_metadata_json(handle));
        public void Dispose() => IbtNative.ibt_close(handle);
    }
}
