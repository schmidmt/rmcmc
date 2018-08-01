use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::path::Path;

pub fn multiple_tries<F: FnMut(usize) -> bool>(
    n_tries: usize,
    mut f: F,
) -> bool {
    for i in 0..n_tries {
        println!("MULTIPLE_TRIES: {}", i);
        if f(i) {
            return true;
        }
    }
    false
}

pub fn write_samples_to_file<T: Display>(
    path: &Path,
    samples: &[T],
) -> io::Result<()> {
    let display = (*path).display();
    let mut file = match File::create(&path) {
        Err(why) => {
            panic!("couldn't create {}: {}", display, why.description())
        }
        Ok(file) => file,
    };

    let string_samples: Vec<String> =
        samples.iter().map(|x| format!("{}", x)).collect();
    let output = string_samples.join("\n");
    match file.write_all(output.as_bytes()) {
        Err(why) => {
            panic!("couldn't write to {}: {}", display, why.description())
        }
        Ok(_) => Ok(()),
    }
}
