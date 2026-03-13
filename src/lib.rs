extern crate self as tangent;

#[doc(hidden)]
pub mod __private {
    pub use crate::nn::scope::{pop_path, push_path};
}

pub mod autodiff;
mod checkpoint;
pub mod losses;
pub mod nn;
pub mod optim;
pub mod tensor;
pub mod utils;
