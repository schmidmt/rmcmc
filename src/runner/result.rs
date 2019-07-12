use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::sync::PoisonError;

/// Errors from Runner Failure
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum RunnerError {
    /// The RW Lock failed (due to panic)
    LockFail,
}

impl Display for RunnerError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "The Runner encountered an error: {:?}", self)
    }
}

/// Result type from Runner
pub type Result<T> = std::result::Result<T, RunnerError>;

impl<T> From<PoisonError<T>> for RunnerError {
    fn from(_: PoisonError<T>) -> Self {
        RunnerError::LockFail
    }
}

impl Error for RunnerError {
    fn description(&self) -> &str {
        match self {
            RunnerError::LockFail => "Panic caused RWLock Failure",
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<RunnerError> for io::Error {
    fn from(e: RunnerError) -> Self {
        io::Error::new(io::ErrorKind::Other, e)
    }
}
