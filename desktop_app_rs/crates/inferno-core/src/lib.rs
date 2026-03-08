pub mod analysis;
pub mod channel;
pub mod error;
pub mod lap;
pub mod profile;
pub mod session;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
