namespace TirePressureCalculator.ViewModels;

public class MainViewModel
{
    public TireCornerViewModel FrontLeft { get; } = new() { Label = "FL" };
    public TireCornerViewModel FrontRight { get; } = new() { Label = "FR" };
    public TireCornerViewModel RearLeft { get; } = new() { Label = "RL" };
    public TireCornerViewModel RearRight { get; } = new() { Label = "RR" };
}
