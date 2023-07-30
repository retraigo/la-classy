#[repr(C)]
#[derive(Debug)]
pub enum LossFunction {
    BinCrossEntropy = 1,
    MSE = 2,
}

#[repr(C)]
#[derive(Debug)]
pub enum Model {
    None = 0,
    Logit = 1,
}

#[repr(C)]
#[derive(Debug)]
pub enum Optimizer {
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
    },
    SGD {
        learning_rate: f64,
    },
    MinibatchSGD {
        learning_rate: f64,
        n_batches: usize,
    },
    GD {
        learning_rate: f64,
    },
}
#[repr(C)]
#[derive(Debug)]
pub enum OptimizerFFI {
    Adam = 1,
    SGD = 2,
    MinibatchSGD = 3,
    GD = 4,
}
