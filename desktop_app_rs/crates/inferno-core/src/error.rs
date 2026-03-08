use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XRK parse error: {0}")]
    Xrk(#[from] libxrk::Error),

    #[error("IBT parse error: {0}")]
    Ibt(#[from] libibt::IbtError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),

    #[error("Missing channel: {0}")]
    MissingChannel(String),

    #[error("No valid laps found")]
    NoValidLaps,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
