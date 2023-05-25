extern crate nalgebra as na;

use na::DVector;

pub enum SvcKernel {
    LINEAR,
    POLY,
    RBF,
}
