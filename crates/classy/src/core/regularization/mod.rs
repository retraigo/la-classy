use nalgebra::DMatrix;

pub struct Regularization {
    l1_strength: f64,
    l2_strength: f64,
}

impl Regularization {
    pub fn from(c: f64, l1_ratio: f64) -> Self {
        if c == 0.0 {
            return Regularization {
                l1_strength: 0.0,
                l2_strength: 0.0
            }
        }
        let strength = 1.0 / c;
        if l1_ratio == 1.0 {
            Regularization {
                l1_strength: strength,
                l2_strength: 0.0,
            }
        } else if l1_ratio == 0.0 {
            Regularization {
                l1_strength: 0.0,
                l2_strength: strength,
            }
        } else {
            let l1_strength = strength * l1_ratio;
            let l2_strength = strength - l1_strength;
            Regularization {
                l1_strength,
                l2_strength,
            }
        }
    }
    pub fn l1_coeff(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        if self.l1_strength == 0.0 {
            DMatrix::zeros(x.nrows(), x.ncols())
        } else {
            self.l1_strength * x.map(|w| w.abs())
        }
    }
    pub fn l2_coeff(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        if self.l2_strength == 0.0 {
            DMatrix::zeros(x.nrows(), x.ncols())
        } else {
            self.l2_strength * x.map(|w| w * w)
        }
    }
    pub fn coeff(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.l1_coeff(x) + self.l2_coeff(x)
    }
}
