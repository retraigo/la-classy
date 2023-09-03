mod sigmoid;
use nalgebra::DVector;

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
    pub fn call_on_all(&self, h: DVector<f64>) -> DVector<f64> {
        match self {
            Self::None => h,
            Self::Sigmoid => h.map(|x| sigmoid(x))
        }
    }
}
