using System.Runtime.InteropServices;
using Apache.Arrow.C;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// P/Invoke surface of <c>libxrk-ffi</c> (crate <c>crates/libxrk-ffi</c> in
/// the libxrk repository, library name <c>xrk_ffi</c>). Strings returned as
/// pointers are owned by the caller and released with <see cref="xrk_string_free"/>.
/// </summary>
internal static unsafe partial class XrkNative
{
    private const string Library = "xrk_ffi";

    static XrkNative()
    {
        NativeLoader.EnsureRegistered(typeof(XrkNative).Assembly);
    }

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial nint xrk_open_path(string path, out nint error);

    [LibraryImport(Library)]
    public static partial nint xrk_open_bytes(byte* data, nuint length, out nint error);

    [LibraryImport(Library)]
    public static partial void xrk_close(nint session);

    [LibraryImport(Library)]
    public static partial nuint xrk_channel_count(nint session);

    [LibraryImport(Library)]
    public static partial nint xrk_channel_name(nint session, nuint index);

    [LibraryImport(Library)]
    public static partial nint xrk_metadata_json(nint session);

    [LibraryImport(Library)]
    public static partial int xrk_channel_stream(nint session, nuint index, CArrowArrayStream* stream);

    [LibraryImport(Library)]
    public static partial int xrk_laps_stream(nint session, CArrowArrayStream* stream);

    [LibraryImport(Library)]
    public static partial void xrk_string_free(nint s);

    [LibraryImport(Library)]
    public static partial nint xrk_version();

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
            xrk_string_free(s);
        }
    }

    /// <summary>The FFI crate's version.</summary>
    public static string Version => Marshal.PtrToStringUTF8(xrk_version()) ?? string.Empty;
}
