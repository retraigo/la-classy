use ndarray::Array2;

pub struct Regularization {
    l1_strength: f32,
    l2_strength: f32,
}

impl Regularization {
    pub fn from(c: f32, l1_ratio: f32) -> Self {
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
    pub fn l1_coeff(&self, x: &Array2<f32>) -> Array2<f32> {
        if self.l1_strength == 0.0 {
            Array2::zeros((x.nrows(), x.ncols()))
        } else {
            self.l1_strength * x.map(|w| w.abs())
        }
    }
    pub fn l2_coeff(&self, x: &Array2<f32>) -> Array2<f32> {
        if self.l2_strength == 0.0 {
            Array2::zeros((x.nrows(), x.ncols()))
        } else {
            self.l2_strength * x.map(|w| w * w)
        }
    }
    pub fn coeff(&self, x: &Array2<f32>) -> Array2<f32> {
        self.l1_coeff(x) + self.l2_coeff(x)
    }
}
