//! Utilities for testing probabilistic programming

use std::panic::{UnwindSafe, catch_unwind};
use std::io;
use std::fmt::Display;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
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


/// Write Vector to file
/// # Parameters
/// * `data` - Data to write to file
/// * `path` - Path of file to be written
pub fn write_vec_file<T: Display>(data: &[T], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;
    for datum in data {
        write!(file, "{}\n", datum)?;
    }
    Ok(())
}
