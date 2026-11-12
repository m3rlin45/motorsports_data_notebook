using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InfernoAnalyzer.Core.Profiles;

namespace InfernoAnalyzer.App.ViewModels;

/// <summary>One channel-name entry (canonical key → logged channel name).</summary>
public sealed partial class ChannelEntryViewModel : ObservableObject
{
    private readonly Action _onChanged;

    /// <summary>Create an entry for <paramref name="key"/>.</summary>
    public ChannelEntryViewModel(string key, string label, string value, Action onChanged)
    {
        Key = key;
        Label = label;
        _value = value;
        _onChanged = onChanged;
    }

    /// <summary>Canonical key, e.g. "shock_fl".</summary>
    public string Key { get; }

    /// <summary>Display label, e.g. "FL Shock".</summary>
    public string Label { get; }

    private string _value;

    /// <summary>The logged channel name.</summary>
    public string Value
    {
        get => _value;
        set
        {
            if (SetProperty(ref _value, value))
            {
                _onChanged();
            }
        }
    }

    /// <summary>Channel names of the session, for completion.</summary>
    [ObservableProperty]
    private IReadOnlyList<string> _suggestions = Array.Empty<string>();

    /// <summary>Whether the entry can be edited (B entries are read-only while synced).</summary>
    [ObservableProperty]
    private bool _isEditable = true;

    /// <summary>Set the value without raising the change callback.</summary>
    public void SetSilently(string value) => SetProperty(ref _value, value, nameof(Value));
}

/// <summary>
/// Channel mapping for one tab, per session: the Python <c>BaseConfigPanel</c>
/// with its A/B entries, "keep B in sync with A" switch and profile fill.
/// </summary>
public partial class ChannelConfigViewModel : ObservableObject
{
    private readonly IReadOnlyList<(string Key, string Label)> _keys;
    private bool _suppress;

    /// <summary>Create entries for <paramref name="keys"/> (canonical key, label), defaulting from <see cref="ChannelKeys.DefaultChannelNames"/>.</summary>
    public ChannelConfigViewModel(IReadOnlyList<(string Key, string Label)> keys)
    {
        _keys = keys;
        foreach (var (key, label) in keys)
        {
            EntriesA.Add(new ChannelEntryViewModel(key, label, Default(key), () => OnEntryChanged("A")));
            EntriesB.Add(new ChannelEntryViewModel(key, label, Default(key), () => OnEntryChanged("B")) { IsEditable = false });
        }
    }

    /// <summary>Raised when any value the analysis depends on changes.</summary>
    public event Action? Changed;

    /// <summary>Session A entries.</summary>
    public ObservableCollection<ChannelEntryViewModel> EntriesA { get; } = new();

    /// <summary>Session B entries (follow A while <see cref="SyncWithA"/>).</summary>
    public ObservableCollection<ChannelEntryViewModel> EntriesB { get; } = new();

    /// <summary>The entries being shown/edited: A or B.</summary>
    public ObservableCollection<ChannelEntryViewModel> Entries => ActiveSession == "B" ? EntriesB : EntriesA;

    /// <summary>"A" or "B".</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Entries))]
    [NotifyPropertyChangedFor(nameof(IsSessionA))]
    [NotifyPropertyChangedFor(nameof(IsSessionB))]
    private string _activeSession = "A";

    /// <summary>Shown once Session B has loaded.</summary>
    [ObservableProperty]
    private bool _hasSessionB;

    /// <summary>Keep B's entries equal to A's.</summary>
    [ObservableProperty]
    private bool _syncWithA = true;

    /// <summary>Radio binding.</summary>
    public bool IsSessionA
    {
        get => ActiveSession == "A";
        set
        {
            if (value)
            {
                ActiveSession = "A";
            }
        }
    }

    /// <summary>Radio binding.</summary>
    public bool IsSessionB
    {
        get => ActiveSession == "B";
        set
        {
            if (value)
            {
                ActiveSession = "B";
            }
        }
    }

    partial void OnSyncWithAChanged(bool value)
    {
        if (value)
        {
            CopyAToB();
        }

        foreach (var entry in EntriesB)
        {
            entry.IsEditable = !value;
        }

        OnSyncChanged(value);
        Changed?.Invoke();
    }

    /// <summary>Hook for subclasses with extra per-session fields.</summary>
    protected virtual void OnSyncChanged(bool synced)
    {
    }

    private static string Default(string key) => ChannelKeys.DefaultChannelNames.TryGetValue(key, out var v) ? v : string.Empty;

    private void OnEntryChanged(string session)
    {
        if (_suppress)
        {
            return;
        }

        if (session == "A" && SyncWithA)
        {
            CopyAToB();
        }

        Changed?.Invoke();
    }

    /// <summary>Copy A's values into B silently.</summary>
    protected void CopyAToB()
    {
        for (var i = 0; i < EntriesA.Count; i++)
        {
            EntriesB[i].SetSilently(EntriesA[i].Value);
        }

        CopyExtrasAToB();
    }

    /// <summary>Hook for subclasses with extra per-session fields.</summary>
    protected virtual void CopyExtrasAToB()
    {
    }

    /// <summary>Fill a session's entries from a profile (silently, then one change event).</summary>
    public void SetFromProfile(VehicleProfile profile, string session = "A")
    {
        _suppress = true;
        foreach (var entry in session == "B" ? EntriesB : EntriesA)
        {
            if (profile.ChannelNames.TryGetValue(entry.Key, out var name))
            {
                entry.SetSilently(name);
            }
        }

        SetExtrasFromProfile(profile, session);
        if (session == "A" && SyncWithA)
        {
            CopyAToB();
        }

        _suppress = false;
        Changed?.Invoke();
    }

    /// <summary>Hook for subclasses: fill their extra fields from the profile.</summary>
    protected virtual void SetExtrasFromProfile(VehicleProfile profile, string session)
    {
    }

    /// <summary>Offer a session's channel names for completion.</summary>
    public void UpdateAvailableChannels(string session, IReadOnlyList<string> names)
    {
        foreach (var entry in session == "B" ? EntriesB : EntriesA)
        {
            entry.Suggestions = names;
        }
    }

    /// <summary>Restore the package defaults for the active session.</summary>
    [RelayCommand]
    public void Reset()
    {
        _suppress = true;
        foreach (var entry in Entries)
        {
            entry.SetSilently(Default(entry.Key));
        }

        if (ActiveSession == "A" && SyncWithA)
        {
            CopyAToB();
        }

        _suppress = false;
        Changed?.Invoke();
    }

    /// <summary>Canonical key → channel name for a session (B follows A while synced).</summary>
    public Dictionary<string, string> GetChannelNames(string session)
    {
        var entries = session == "B" && !SyncWithA ? EntriesB : EntriesA;
        return entries.ToDictionary(e => e.Key, e => e.Value.Trim(), StringComparer.Ordinal);
    }
}
