use ndarray::Array2;

pub mod crossentropy;
pub mod hinge;
pub mod mean;

pub enum LossFunction {
    BinCrossEntropy,
    CrossEntropy,
    Hinge,
    Huber(f32),
    MAE,
    MSE,
    SmoothedHinge,
    Tukey(f32),
}

impl LossFunction {
    pub fn loss(&self, y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy(&y, &y1),
            LossFunction::CrossEntropy => crossentropy::cross_entropy(&y, &y1),
            LossFunction::Hinge => hinge::hinge(y, y1),
            LossFunction::Huber(delta) => mean::huber(y, y1, *delta),
            LossFunction::MAE => mean::mae(&y, &y1),
            LossFunction::MSE => mean::mse(&y, &y1),
            LossFunction::SmoothedHinge => hinge::smooth_hinge(y, y1),
            LossFunction::Tukey(c) => mean::tukey(y, y1, *c),
        }
    }
    pub fn loss_d(&self, y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy_d(&y, &y1),
            LossFunction::CrossEntropy => crossentropy::cross_entropy_d(&y, &y1),
            LossFunction::Hinge => hinge::hinge_d(y, y1),
            LossFunction::Huber(delta) => mean::huber_d(y, y1, *delta),
            LossFunction::MAE => mean::mae_d(&y, &y1),
            LossFunction::MSE => mean::mse_d(&y, &y1),
            LossFunction::SmoothedHinge => hinge::smooth_hinge_d(y, y1),
            LossFunction::Tukey(c) => mean::tukey_d(y, y1, *c),
        }
    }
}
