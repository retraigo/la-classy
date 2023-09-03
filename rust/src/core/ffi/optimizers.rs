use crate::core::optimizers::{OptimizerConfig, Optimizer};

#[no_mangle]
pub unsafe extern "C" fn adam_optimizer(
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    input_size: usize,
) -> isize {
    let config = OptimizerConfig::Adam { beta1, beta2, epsilon };
    let opt = Optimizer::from(config, input_size);
    std::mem::transmute::<Box<Optimizer>, isize>(std::boxed::Box::new(opt))
}

#[no_mangle]
pub unsafe extern "C" fn no_optimizer() -> isize {
    let config = OptimizerConfig::None;
    let opt = Optimizer::from(config, 0);
    std::mem::transmute::<Box<Optimizer>, isize>(std::boxed::Box::new(opt))
}