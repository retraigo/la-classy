use nalgebra::{DMatrix, DVector};
use rand::Rng;

use crate::{
    core::{
        loss::{binary_cross_entropy, cross_entropy, mean_absolute_error, mean_squared_error},
        scheduler::get_learning_rate, functions::sigmoid,
    },
    types::{AdamOptimizerConfig, LossFunction, Model, ModelConfig, Scheduler},
};

pub struct AdamOptimizer {
    config: ModelConfig,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    scheduler: Scheduler,
    m: DVector<f64>,
    v: DVector<f64>,
    t: usize,
}

impl AdamOptimizer {
    pub fn new(
        config: AdamOptimizerConfig,
        n_features: usize,
        learning_rate: f64,
        model: ModelConfig,
        scheduler: Scheduler,
    ) -> AdamOptimizer {
        let mut m = DVector::zeros(n_features + if model.fit_intercept { 1 } else { 0 });
        let mut v = DVector::zeros(n_features + if model.fit_intercept { 1 } else { 0 });
        AdamOptimizer {
            config: model,
            learning_rate,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            scheduler,
            m,
            v,
            t: 0,
        }
    }
    pub fn train(&self, data: DMatrix<f64>, targets: DVector<f64>) -> (DVector<f64>, f64) {
        let mut eta = self.learning_rate;
        let mut rng = rand::thread_rng();
        let mut weights = DVector::from_element(
            data.ncols() + if self.config.fit_intercept { 1 } else { 0 },
            1.0,
        );
        for i in 0..self.config.epochs {
            eta = get_learning_rate(&self.scheduler, eta, i, self.learning_rate);
            if i % 100 == 0 {
                if !self.config.silent {
                    let mut h = data * &weights;

                    match self.config.model {
                        Model::None => (),
                        Model::Logit => h.apply(|x| {
                            let res = sigmoid(*x);
                            *x = res;
                        }),
                    };

                    let error: f64 = match self.config.loss {
                        LossFunction::CrossEntropy => cross_entropy(&targets, &h),
                        LossFunction::BinCrossEntropy => binary_cross_entropy(&targets, &h),
                        LossFunction::MAE => mean_absolute_error(&targets, &h),
                        LossFunction::MSE => mean_squared_error(&targets, &h),
                    };
                    println!("Epoch <{}: Current Errors {}", i, error);
                }
            }

            let beta1_pow_t = self.beta1.powi(self.t as i32);
            let beta2_pow_t = self.beta2.powi(self.t as i32);

            let inverse_n = 1.0 / data.nrows() as f64;
            let mut h = data * &weights;
            match self.config.model {
                Model::None => (),
                Model::Logit => h.apply(|x| {
                    let res = sigmoid(*x);
                    *x = res;
                }),
            };
            let errors = h - targets;
            let weight_updates = &data.transpose() * &errors * inverse_n;

            let l1 = c * &weights.map(|w| if w >= 0.0 { 1.0 } else { -1.0 });

            weight_m *= self.beta1;
            weight_m += (1.0 - self.beta1) * (&weight_updates + &l1);

            weight_v *= self.beta2;
            weight_v += (1.0 - self.beta2)
                * (&weight_updates.component_mul(&weight_updates) + &l1.component_mul(&l1));

            let m_hat = &weight_m / (1.0 - beta1_pow_t);
            let v_hat = &weight_v / (1.0 - beta2_pow_t);

            let updates =
                &m_hat.zip_map(&v_hat.map(|x| x.sqrt()).add_scalar(epsilon), |x, y| x / y);
            weights = weights - updates * eta;

            if self.config.fit_intercept {
                let intercept_updates = errors.sum() * inverse_n;
                let intercept_l1 = c * if intercept >= 0.0 { 1.0 } else { -1.0 };

                intercept_m *= self.beta1;
                intercept_m += (1.0 - self.beta1) * intercept_updates + intercept_l1;

                intercept_v *= self.beta2;
                intercept_v += (1.0 - self.beta2) * (intercept_updates + intercept_l1).powi(2);

                let intercept_m_hat = intercept_m / (1.0 - beta1_pow_t);
                let intercept_v_hat = intercept_v / (1.0 - beta2_pow_t);

                intercept -= (eta / (intercept_v_hat.sqrt() + self.epsilon)) * intercept_m_hat;
            }

            self.t += 1;
        }

        (targets, 1.0)
    }
}
