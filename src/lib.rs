#![feature(associated_type_defaults)]
#![feature(test)]

#[cfg(feature = "serde_support")]
#[macro_use]
extern crate serde_derive;

extern crate alga;
extern crate typenum;
extern crate nalgebra;
extern crate rand;
extern crate reduce;
extern crate rv;

#[macro_use]
pub mod lens;
pub mod common;
pub mod likelihood;
pub mod parameter;
pub mod runner;
pub mod statistics;
pub mod steppers;
pub mod summary;
pub mod utils;
