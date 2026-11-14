using System.Security.Cryptography;
using System.Text;
using InfernoAnalyzer.App;

namespace InfernoAnalyzer.Tests;

/// <summary>The self-extracting build's library cache: what gets written, when, and where.</summary>
public sealed class NativeBundleTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "inferno-bundle-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, byte[]> _content = new()
    {
        ["libSkiaSharp.dll"] = Encoding.UTF8.GetBytes("skia"),
        ["av_libglesv2.dll"] = Encoding.UTF8.GetBytes("angle"),
    };

    private int _opened;

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    private IReadOnlyList<NativeBundle.Entry> Entries() => _content
        .Select(pair => new NativeBundle.Entry(pair.Key, Convert.ToHexStringLower(SHA256.HashData(pair.Value))))
        .ToList();

    private Stream Open(string name)
    {
        _opened++;
        return new MemoryStream(_content[name]);
    }

    [Fact]
    public void Libraries_are_written_once_and_reused()
    {
        var first = NativeBundle.Extract(Entries(), Open, new[] { _root });

        Assert.Equal(_root, Path.GetDirectoryName(first));
        Assert.Equal(_content["libSkiaSharp.dll"], File.ReadAllBytes(Path.Combine(first, "libSkiaSharp.dll")));
        Assert.Equal(_content["av_libglesv2.dll"], File.ReadAllBytes(Path.Combine(first, "av_libglesv2.dll")));
        Assert.Equal(2, _opened);
        Assert.Equal(2, Directory.GetFiles(first).Length); // no temporary files left behind

        var second = NativeBundle.Extract(Entries(), Open, new[] { _root });

        Assert.Equal(first, second);
        Assert.Equal(2, _opened); // nothing rewritten
    }

    [Fact]
    public void An_altered_or_missing_library_is_replaced()
    {
        var directory = NativeBundle.Extract(Entries(), Open, new[] { _root });
        File.WriteAllText(Path.Combine(directory, "libSkiaSharp.dll"), "tampered");
        File.Delete(Path.Combine(directory, "av_libglesv2.dll"));

        NativeBundle.Extract(Entries(), Open, new[] { _root });

        Assert.Equal(_content["libSkiaSharp.dll"], File.ReadAllBytes(Path.Combine(directory, "libSkiaSharp.dll")));
        Assert.Equal(_content["av_libglesv2.dll"], File.ReadAllBytes(Path.Combine(directory, "av_libglesv2.dll")));
    }

    [Fact]
    public void Different_libraries_get_their_own_folder_and_the_old_one_goes()
    {
        var old = NativeBundle.Extract(Entries(), Open, new[] { _root });
        _content["libSkiaSharp.dll"] = Encoding.UTF8.GetBytes("skia, next version");

        var current = NativeBundle.Extract(Entries(), Open, new[] { _root });

        Assert.NotEqual(old, current);
        Assert.False(Directory.Exists(old));
        Assert.Equal(_content["libSkiaSharp.dll"], File.ReadAllBytes(Path.Combine(current, "libSkiaSharp.dll")));
    }

    [Fact]
    public void A_root_that_cannot_be_used_falls_through_to_the_next()
    {
        Directory.CreateDirectory(_root);
        var blocked = Path.Combine(_root, "a-file");
        File.WriteAllText(blocked, "not a folder");
        var fallback = Path.Combine(_root, "fallback");

        var directory = NativeBundle.Extract(Entries(), Open, new[] { blocked, fallback });

        Assert.Equal(fallback, Path.GetDirectoryName(directory));
        Assert.Throws<AggregateException>(() => NativeBundle.Extract(Entries(), Open, new[] { blocked }));
    }

    [Fact]
    public void Manifest_lines_and_library_names_are_read_as_the_build_and_the_imports_write_them()
    {
        var entries = NativeBundle.ParseManifest("libSkiaSharp.dll 66C856EA\r\nav_libglesv2.dll 3731e7f4\n");

        Assert.Equal(new[] { new NativeBundle.Entry("libSkiaSharp.dll", "66c856ea"), new NativeBundle.Entry("av_libglesv2.dll", "3731e7f4") }, entries);
        Assert.Equal("av_libGLESv2", NativeBundle.LibraryKey("av_libGLESv2.dll"));
        Assert.Equal("libSkiaSharp", NativeBundle.LibraryKey("libSkiaSharp"));
        Assert.Equal("libHarfBuzzSharp", NativeBundle.LibraryKey("libHarfBuzzSharp.so"));
    }
}
