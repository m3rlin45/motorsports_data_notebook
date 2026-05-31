pub mod channel;
pub mod error;
pub mod lap;
pub mod session;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
