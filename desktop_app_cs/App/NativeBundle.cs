using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace InfernoAnalyzer.App;

/// <summary>
/// The graphics libraries of a self-extracting build. Avalonia's SkiaSharp, HarfBuzzSharp and (on
/// Windows) ANGLE exist as shared libraries only, and the operating system loads a shared library
/// from a file, so a build published with <c>InfernoSelfExtract</c> carries them as embedded
/// resources (see the project file) and, before anything draws, writes them to a per-user cache
/// folder, loads them from there and answers the importing assemblies' library lookups with them
/// (SkiaSharp looks beside the executable only, so being loaded already is not enough). The cache
/// folder is named after the libraries' hashes: a new version of the app with the same libraries
/// reuses it, and every start checks the files against the hashes, so a damaged or altered file is
/// replaced.
/// </summary>
internal static class NativeBundle
{
    private const string ResourcePrefix = "native/";
    private const string ManifestName = ResourcePrefix + "manifest";

    /// <summary>One bundled library: its file name and the SHA-256 of its content, in hex.</summary>
    internal readonly record struct Entry(string FileName, string Sha256);

    /// <summary>
    /// Extract and load the bundled libraries, if this build carries any, and serve them to the
    /// P/Invokes of <paramref name="importers"/> (the assemblies that import them). Call before
    /// Avalonia starts.
    /// </summary>
    public static void Load(params Assembly[] importers)
    {
        var assembly = typeof(NativeBundle).Assembly;
        using var manifest = assembly.GetManifestResourceStream(ManifestName);
        if (manifest is null)
        {
            return; // not a self-extracting build: the libraries are files beside the executable
        }

        var entries = ParseManifest(new StreamReader(manifest).ReadToEnd());
        var directory = Extract(entries, name => OpenResource(assembly, name), CacheRoots());
        var loaded = entries.ToDictionary(
            entry => LibraryKey(entry.FileName),
            entry => NativeLibrary.Load(Path.Combine(directory, entry.FileName)),
            StringComparer.OrdinalIgnoreCase);
        foreach (var importer in importers)
        {
            // Zero sends any other library (user32, libX11, ...) down the default lookup.
            NativeLibrary.SetDllImportResolver(importer, (name, _, _) => loaded.GetValueOrDefault(LibraryKey(name)));
        }
    }

    /// <summary>A library name as imports and files agree on it: no extension, any case ("av_libGLESv2.dll" and "av_libglesv2").</summary>
    internal static string LibraryKey(string name) =>
        name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".so", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".dylib", StringComparison.OrdinalIgnoreCase)
            ? Path.GetFileNameWithoutExtension(name)
            : name;

    /// <summary>The manifest: one "file-name sha256" line per library.</summary>
    internal static IReadOnlyList<Entry> ParseManifest(string text) => text
        .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(line => line.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        .Select(parts => new Entry(parts[0], parts[1].ToLowerInvariant()))
        .ToList();

    /// <summary>
    /// Make sure every library is in a folder under one of <paramref name="roots"/> with the content
    /// the manifest names, writing the ones that are missing or differ, and return that folder. The
    /// roots are tried in order; the first that can be written to is used.
    /// </summary>
    internal static string Extract(IReadOnlyList<Entry> entries, Func<string, Stream> open, IEnumerable<string> roots)
    {
        var key = Convert.ToHexStringLower(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(string.Join('\n', entries))))[..16];
        var failures = new List<Exception>();
        foreach (var root in roots)
        {
            var directory = Path.Combine(root, key);
            try
            {
                Directory.CreateDirectory(directory);
                foreach (var entry in entries)
                {
                    var path = Path.Combine(directory, entry.FileName);
                    if (!Matches(path, entry.Sha256))
                    {
                        Write(path, open(entry.FileName));
                    }
                }

                RemoveOtherVersions(root, key);
                return directory;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                failures.Add(e);
            }
        }

        throw new AggregateException("The graphics libraries could not be unpacked to any cache folder.", failures);
    }

    /// <summary>Where the libraries may be unpacked, best first: the per-user application data folder, then the temporary folder.</summary>
    private static IEnumerable<string> CacheRoots()
    {
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (!string.IsNullOrEmpty(local))
        {
            yield return Path.Combine(local, "InfernoAnalyzer", "native");
        }

        yield return Path.Combine(Path.GetTempPath(), "InfernoAnalyzer", "native");
    }

    private static Stream OpenResource(Assembly assembly, string fileName) =>
        assembly.GetManifestResourceStream(ResourcePrefix + fileName)
        ?? throw new InvalidOperationException($"The build lists {fileName} as bundled but does not contain it.");

    private static bool Matches(string path, string sha256)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(file)) == sha256;
    }

    /// <summary>Write through a temporary file and a rename, so a second instance starting at the same moment never sees half a library.</summary>
    private static void Write(string path, Stream content)
    {
        var temporary = $"{path}.{Environment.ProcessId}.tmp";
        try
        {
            using (content)
            using (var file = File.Create(temporary))
            {
                content.CopyTo(file);
            }

            File.Move(temporary, path, overwrite: true);
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    /// <summary>Drop the folders of other library versions. Best effort: one in use by a running instance stays.</summary>
    private static void RemoveOtherVersions(string root, string key)
    {
        foreach (var other in Directory.EnumerateDirectories(root))
        {
            if (Path.GetFileName(other) == key)
            {
                continue;
            }

            try
            {
                Directory.Delete(other, recursive: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // Still loaded by another instance; the next start tries again.
            }
        }
    }
}
