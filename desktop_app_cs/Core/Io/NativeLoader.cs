using System.Runtime.InteropServices;

namespace InfernoAnalyzer.Core.Io;

/// <summary>
/// Finds the parser FFI libraries. They are copied to
/// <c>runtimes/&lt;rid&gt;/native/</c> under the application directory at
/// build time and sit next to the executable after a RID-specific publish;
/// the default probing only knows about the latter.
/// </summary>
internal static class NativeLoader
{
    private static readonly object Gate = new();
    private static bool _registered;

    /// <summary>Register the resolver for <paramref name="assembly"/> once.</summary>
    public static void EnsureRegistered(System.Reflection.Assembly assembly)
    {
        lock (Gate)
        {
            if (_registered)
            {
                return;
            }

            NativeLibrary.SetDllImportResolver(assembly, Resolve);
            _registered = true;
        }
    }

    /// <summary>The portable runtime identifier, e.g. <c>linux-x64</c>.</summary>
    public static string RuntimeIdentifier
    {
        get
        {
            var os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
            var arch = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "arm64",
                Architecture.X86 => "x86",
                _ => "x64",
            };
            return $"{os}-{arch}";
        }
    }

    /// <summary>Candidate file names for a library called <paramref name="name"/> on this OS.</summary>
    public static IEnumerable<string> FileNames(string name)
    {
        if (OperatingSystem.IsWindows())
        {
            yield return $"{name}.dll";
        }
        else if (OperatingSystem.IsMacOS())
        {
            yield return $"lib{name}.dylib";
            yield return $"{name}.dylib";
        }
        else
        {
            yield return $"lib{name}.so";
            yield return $"{name}.so";
        }
    }

    /// <summary>Directories searched, in order.</summary>
    public static IEnumerable<string> SearchDirectories()
    {
        var baseDir = AppContext.BaseDirectory;
        yield return Path.Combine(baseDir, "runtimes", RuntimeIdentifier, "native");
        yield return baseDir;
    }

    private static nint Resolve(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        foreach (var dir in SearchDirectories())
        {
            foreach (var file in FileNames(libraryName))
            {
                var path = Path.Combine(dir, file);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out var handle))
                {
                    return handle;
                }
            }
        }

        return nint.Zero; // fall back to the runtime's default probing
    }
}
