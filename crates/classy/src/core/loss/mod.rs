use nalgebra::DVector;
pub mod crossentropy;
pub mod mean;

pub enum LossFunction {
    BinCrossEntropy,
    MAE,
    MSE
}

impl LossFunction {
    pub fn loss(&self, y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy(&y, &y1),
            LossFunction::MAE => mean::mae(&y, &y1),
            LossFunction::MSE => mean::mse(&y, &y1),
        }
    }
    pub fn loss_d(&self, y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
        match self {
            LossFunction::BinCrossEntropy => crossentropy::bin_cross_entropy_d(&y, &y1),
            LossFunction::MAE => mean::mae_d(&y, &y1),
            LossFunction::MSE => mean::mse_d(&y, &y1),
        }
    }
}
