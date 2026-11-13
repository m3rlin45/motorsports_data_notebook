using Avalonia.Automation.Peers;
using Avalonia.Controls;

namespace InfernoAnalyzer.App.Controls;

/// <summary>
/// Automation peer of a custom-drawn control: it has no child controls for a
/// screen reader to walk, so it at least announces itself by name, with a
/// summary of what it shows, and takes part in keyboard navigation.
/// </summary>
internal sealed class CustomDrawnPeer : ControlAutomationPeer
{
    private readonly AutomationControlType _type;
    private readonly Func<string> _describe;

    public CustomDrawnPeer(Control owner, AutomationControlType type, Func<string> describe)
        : base(owner)
    {
        _type = type;
        _describe = describe;
    }

    protected override AutomationControlType GetAutomationControlTypeCore() => _type;

    protected override string GetNameCore() => _describe();

    protected override bool IsContentElementCore() => true;

    protected override bool IsControlElementCore() => true;
}
