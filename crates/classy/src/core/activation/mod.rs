use nalgebra::DMatrix;

pub enum Activation {
    Linear,
    Sigmoid,
    Tanh,
    Softmax,
}

impl Activation {
    pub fn call_on_all(&self, h: DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Self::Linear => h,
            Self::Sigmoid => h.map(|x| sigmoid(x)),
            Self::Tanh => h.map(|x| x.tanh()),
            Self::Softmax => {
                let mut res = DMatrix::zeros(h.nrows(), h.ncols());
                for (mut res_row, h_row) in res.row_iter_mut().zip(h.row_iter()) {
                    let exp_values = h_row.map(|v| v.exp());
                    let sum_exp: f64 = exp_values.iter().sum();
                    res_row.copy_from(&(exp_values / sum_exp));
                }
                res
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
