using System.Runtime.InteropServices.JavaScript;

namespace TirePressureCalculator;

internal sealed partial class LocalStorageSettingsStorage : ISettingsStorage
{
    private const string Key = "tire-pressure-calculator/settings";

    public string? Read() => GetItem(Key);

    public void Write(string contents) => SetItem(Key, contents);

    [JSImport("globalThis.localStorage.getItem")]
    private static partial string? GetItem(string key);

    [JSImport("globalThis.localStorage.setItem")]
    private static partial void SetItem(string key, string value);
}
