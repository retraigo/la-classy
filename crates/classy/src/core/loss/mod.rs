use nalgebra::DMatrix;

pub mod crossentropy;
pub mod mean;
pub mod hinge;

pub enum LossFunction {
    BinCrossEntropy,
    CrossEntropy,
    Hinge,
    MAE,
    MSE
}

impl LossFunction {
    pub fn loss(&self, y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy(&y, &y1),
            LossFunction::CrossEntropy => crossentropy::cross_entropy(&y, &y1),
            LossFunction::Hinge => hinge::hinge(y, y1),
            LossFunction::MAE => mean::mae(&y, &y1),
            LossFunction::MSE => mean::mse(&y, &y1),
        }
    }
    pub fn loss_d(&self, y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy_d(&y, &y1),
            LossFunction::CrossEntropy => crossentropy::cross_entropy_d(&y, &y1),
            LossFunction::Hinge => hinge::hinge_d(y, y1),
            LossFunction::MAE => mean::mae_d(&y, &y1),
            LossFunction::MSE => mean::mse_d(&y, &y1),
        }
    }
}
