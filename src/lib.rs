#![feature(associated_type_defaults)]
#![feature(test)]

#[cfg(feature = "serde_support")]
extern crate serde_derive;

extern crate alga;
extern crate typenum;
extern crate nalgebra;
extern crate rand;
extern crate reduce;
extern crate rv;
extern crate rayon;
extern crate num;
extern crate itertools;

#[macro_use] mod lens;
pub use lens::*;

mod parameter;
pub use parameter::*;

mod runner;
pub use runner::*;

pub mod steppers;
pub mod utils;
pub mod diagnostics;
