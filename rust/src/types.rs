use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub epochs: usize,
    pub silent: bool,
    pub learning_rate: f64,
    pub fit_intercept: bool,
    pub model: Model,
    pub loss: LossFunction,
    pub optimizer: Optimizer,
    pub scheduler: Scheduler,
    pub n_batches: usize,
    pub c: f64,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum LossFunction {
    MAE,
    MSE,
    CrossEntropy,
    BinCrossEntropy,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    None,
    Logit,
}
#[repr(C)]
pub enum Solver {
    OLS = 0,
    GD = 1,
    SGD = 2,
    Minibatch = 3,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    None,
    Adam(AdamOptimizerConfig),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct AdamOptimizerConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MinibatchSGDOptimizerConfig {
    pub n_batches: usize,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type", content = "config")]
#[serde(rename_all = "lowercase")]
pub enum Scheduler {
    None,
    ExponentialAnnealer {
        // Rate of decay
        rate: f64,
    },
    LinearAnnealer {
        // Rate of decay
        rate: f64,
    },
    DecayScheduler {
        // Rate of decay
        rate: f64,
        // Number of epochs for decay
        step_size: usize,
    },
    OneCycleScheduler {
        // Max allowed learning rate
        max_lr: f64,
        // Number of steps in one cycle
        cycle_steps: usize,
    },
}