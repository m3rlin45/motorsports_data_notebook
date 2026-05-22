using System;
using System.IO;
using System.Reflection;
using System.Text.Json;

namespace TirePressureCalculator.Services.Modeling;

/// <summary>
/// Loads the bundled <c>tire_model.json</c> embedded resource (see the
/// MSBuild <c>EmbeddedResource</c> entry in <c>TirePressureCalculator.Core.csproj</c>).
/// </summary>
public static class TireModelLoader
{
    private const string EmbeddedName = "tire_model.json";

    public static TireModel LoadEmbedded()
    {
        var asm = typeof(TireModelLoader).Assembly;
        return LoadEmbedded(asm);
    }

    internal static TireModel LoadEmbedded(Assembly assembly)
    {
        using var stream = assembly.GetManifestResourceStream(EmbeddedName)
            ?? throw new InvalidOperationException(
                $"Embedded tire model resource '{EmbeddedName}' not found in {assembly.FullName}. " +
                "Check the EmbeddedResource entry in TirePressureCalculator.Core.csproj.");
        return LoadFromStream(stream);
    }

    public static TireModel LoadFromFile(string path)
    {
        using var stream = File.OpenRead(path);
        return LoadFromStream(stream);
    }

    public static TireModel LoadFromStream(Stream stream)
    {
        var dto = JsonSerializer.Deserialize(stream, TireModelJsonContext.Default.TireModelDto)
            ?? throw new InvalidDataException("tire_model.json deserialized to null.");
        return new TireModel(dto);
    }
}
