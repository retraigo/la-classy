use crate::core::loss::LossFunction;

#[no_mangle]
pub unsafe extern "C" fn logit_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::BinCrossEntropy))
}

#[no_mangle]
pub unsafe extern "C" fn mae_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::MAE))
}

#[no_mangle]
pub unsafe extern "C" fn mse_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::MSE))
}