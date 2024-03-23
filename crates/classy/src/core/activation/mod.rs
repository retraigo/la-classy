use ndarray::{Array2, Axis};

pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Softmax,
}

impl Activation {
    pub fn call_on_all(&self, h: Array2<f32>) -> Array2<f32> {
        match self {
            Self::Linear => h,
            Self::Sigmoid => h.map(|x| sigmoid(*x)),
            Self::Tanh => h.map(|x| x.tanh()),
            Self::Softmax => {
                let mut res = Array2::zeros((h.nrows(), h.ncols()));
                for (mut res_row, h_row) in res.axis_iter_mut(Axis(0)).zip(h.axis_iter(Axis(0))) {
                    let exp_values = h_row.map(|v| v.exp());
                    let sum_exp: f32 = exp_values.sum();
                    res_row.assign(&(exp_values / sum_exp));
                }
                res
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
