mod sigmoid;
use ndarray::Array1;

use self::sigmoid::sigmoid;

pub enum Activation {
    None,
    Sigmoid,
}

impl Activation {
    pub fn call(&self, h: f64) -> f64 {
        match self {
            Self::None => h,
            Self::Sigmoid => sigmoid(h),
        }
    }
    pub fn call_on_all(&self, h: Array1<f64>) -> Array1<f64> {
        match self {
            Self::None => h,
            Self::Sigmoid => h.map(|x| sigmoid(*x))
        }
    }
}
