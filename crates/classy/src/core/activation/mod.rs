use nalgebra::DVector;

pub enum Activation {
    Linear,
    Sigmoid,
    Tanh
}

impl Activation {
    pub fn call(&self, h: f64) -> f64 {
        match self {
            Self::Linear => h,
            Self::Sigmoid => sigmoid(h),
            Self::Tanh => h.tanh()
        }
    }
    pub fn call_on_all(&self, h: DVector<f64>) -> DVector<f64> {
        match self {
            Self::Linear => h,
            Self::Sigmoid => h.map(|x| sigmoid(x)),
            Self::Tanh => h.map(|x| x.tanh())
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}