use egui::Color32;

/// Viridis colormap sample points (16 samples from 0.0 to 1.0).
const VIRIDIS: [(f64, [u8; 3]); 16] = [
    (0.000, [68, 1, 84]),
    (0.067, [72, 26, 108]),
    (0.133, [71, 47, 126]),
    (0.200, [65, 68, 135]),
    (0.267, [57, 86, 140]),
    (0.333, [49, 104, 142]),
    (0.400, [42, 120, 142]),
    (0.467, [35, 137, 142]),
    (0.533, [31, 152, 139]),
    (0.600, [34, 168, 132]),
    (0.667, [53, 183, 121]),
    (0.733, [90, 197, 101]),
    (0.800, [137, 208, 80]),
    (0.867, [189, 216, 52]),
    (0.933, [229, 228, 32]),
    (1.000, [253, 231, 37]),
];

/// Look up a viridis color by normalized value [0, 1].
pub fn viridis(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0);

    // Find the bounding sample points
    let mut lo = 0;
    for (i, entry) in VIRIDIS.iter().enumerate().skip(1) {
        if entry.0 > t {
            break;
        }
        lo = i;
    }
    let hi = (lo + 1).min(VIRIDIS.len() - 1);

    if lo == hi {
        let [r, g, b] = VIRIDIS[lo].1;
        return Color32::from_rgb(r, g, b);
    }

    // Linear interpolation
    let frac = (t - VIRIDIS[lo].0) / (VIRIDIS[hi].0 - VIRIDIS[lo].0);
    let lerp = |a: u8, b: u8| -> u8 {
        let v = a as f64 + (b as f64 - a as f64) * frac;
        v.round() as u8
    };

    let [r0, g0, b0] = VIRIDIS[lo].1;
    let [r1, g1, b1] = VIRIDIS[hi].1;
    Color32::from_rgb(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

/// Map a viridis color for a given lap index out of total laps.
pub fn viridis_for_lap(lap_index: usize, total_laps: usize) -> Color32 {
    if total_laps <= 1 {
        return viridis(0.5);
    }
    let t = lap_index as f64 / (total_laps - 1) as f64;
    viridis(t)
}

/// Steelblue base color.
pub const STEELBLUE: Color32 = Color32::from_rgb(70, 130, 180);

/// Gold color for opportunity highlights.
pub const GOLD: Color32 = Color32::from_rgb(218, 165, 32);

/// Dark orange for braking point boxes.
pub const DARKORANGE: Color32 = Color32::from_rgb(255, 140, 0);

/// Opportunity highlight band color (gold with transparency).
pub const OPPORTUNITY_BAND: Color32 = Color32::from_rgba_premultiplied(218, 165, 32, 30);

/// Interpolate between steelblue and gold by a normalized opportunity score [0, 1].
pub fn opportunity_gradient(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| -> u8 {
        let v = a as f64 + (b as f64 - a as f64) * t;
        v.round() as u8
    };
    Color32::from_rgb(
        lerp(70, 218),  // steelblue.r → gold.r
        lerp(130, 165), // steelblue.g → gold.g
        lerp(180, 32),  // steelblue.b → gold.b
    )
}

/// Segment type colors for the track map.
pub mod segment {
    use egui::Color32;

    pub const BRAKING: Color32 = Color32::from_rgb(220, 50, 50);
    pub const CORNER: Color32 = Color32::from_rgb(230, 160, 50);
    pub const ACCELERATION: Color32 = Color32::from_rgb(50, 180, 80);
    pub const TRACK_BASE: Color32 = Color32::from_rgb(100, 100, 100);
    pub const APEX_MARKER: Color32 = Color32::from_rgb(139, 0, 0);
}

/// Cyan for braking start VLine.
pub const BRAKING_START_COLOR: Color32 = Color32::from_rgb(0, 200, 200);

/// Yellow for corner entry/exit VLines.
pub const CORNER_BOUNDARY_COLOR: Color32 = Color32::from_rgb(230, 230, 0);

/// Red for apex VLine.
pub const APEX_VLINE_COLOR: Color32 = Color32::from_rgb(220, 50, 50);
