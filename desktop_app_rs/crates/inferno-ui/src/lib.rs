pub mod charts;
pub mod theme;
pub mod widgets;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
