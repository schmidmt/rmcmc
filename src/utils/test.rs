//! Utilities for testing probabilistic programming

use std::panic::{UnwindSafe, catch_unwind};
use log::warn;

/// Try to run a closure up to `limit` times or once without an assertion failure.
///
/// # Parameters
/// * `limit` - Maximum number of times to run the closure.
/// * `f` - Closure to run.
pub fn assert_some_failures<F>(limit: usize, f: F)
where
    F: Fn() -> () + UnwindSafe + Copy,
{
    for _ in 0..limit {
        match catch_unwind(f) {
            Ok(()) => return,
            Err(err) => {
                warn!("Assert_some_failures: unwound with {:?}", err);
            },
        }
    }
    panic!("assert_some_failures: Reached limit, too many assertion failures.");
}