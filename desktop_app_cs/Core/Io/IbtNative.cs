using System.Runtime.InteropServices;
using Apache.Arrow.C;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// P/Invoke surface of <c>libibt-ffi</c> (crate <c>crates/libibt-ffi</c> in
/// the libibt repository, library name <c>ibt_ffi</c>). Strings returned as
/// pointers are owned by the caller and released with <see cref="ibt_string_free"/>.
/// </summary>
internal static unsafe partial class IbtNative
{
    private const string Library = "ibt_ffi";

    static IbtNative()
    {
        NativeLoader.EnsureRegistered(typeof(IbtNative).Assembly);
    }

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial nint ibt_open_path(string path, out nint error);

    [LibraryImport(Library)]
    public static partial nint ibt_open_bytes(byte* data, nuint length, out nint error);

    [LibraryImport(Library)]
    public static partial void ibt_close(nint session);

    [LibraryImport(Library)]
    public static partial nuint ibt_channel_count(nint session);

    [LibraryImport(Library)]
    public static partial nint ibt_channel_name(nint session, nuint index);

    [LibraryImport(Library)]
    public static partial nint ibt_metadata_json(nint session);

    [LibraryImport(Library)]
    public static partial int ibt_channel_stream(nint session, nuint index, CArrowArrayStream* stream);

    [LibraryImport(Library)]
    public static partial int ibt_laps_stream(nint session, CArrowArrayStream* stream);

    [LibraryImport(Library)]
    public static partial void ibt_string_free(nint s);

    [LibraryImport(Library)]
    public static partial nint ibt_version();

    /// <summary>Copy an owned UTF-8 string out of native memory and free it.</summary>
    public static string TakeString(nint s)
    {
        if (s == nint.Zero)
        {
            return string.Empty;
        }

        try
        {
            return Marshal.PtrToStringUTF8(s) ?? string.Empty;
        }
        finally
        {
            ibt_string_free(s);
        }
    }

    /// <summary>The FFI crate's version.</summary>
    public static string Version => Marshal.PtrToStringUTF8(ibt_version()) ?? string.Empty;
}
