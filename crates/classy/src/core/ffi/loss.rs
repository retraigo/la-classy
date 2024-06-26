use crate::core::loss::LossFunction;

#[no_mangle]
pub unsafe extern "C" fn logit_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::BinCrossEntropy))
}

#[no_mangle]
pub unsafe extern "C" fn crossentropy_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::CrossEntropy))
}

#[no_mangle]
pub unsafe extern "C" fn hinge_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::Hinge))
}

#[no_mangle]
pub unsafe extern "C" fn huber_loss(delta: f32) -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::Huber(delta)))
}

#[no_mangle]
pub unsafe extern "C" fn mae_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::MAE))
}

#[no_mangle]
pub unsafe extern "C" fn mse_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::MSE))
}

#[no_mangle]
pub unsafe extern "C" fn smooth_hinge_loss() -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::SmoothedHinge))
}

#[no_mangle]
pub unsafe extern "C" fn tukey_loss(c: f32) -> isize {
    std::mem::transmute::<Box<LossFunction>, isize>(std::boxed::Box::new(LossFunction::Tukey(c)))
}