pub enum Scheduler {
    None,
    LinearDecay {
        // Rate of decay
        rate: f64,
        // Number of epochs for decay
        step_size: usize,
    },
    ExponentialDecay {
        // Rate of decay
        rate: f64,
        // Number of epochs for decay
        step_size: usize,
    },
    OneCycle {
        // Max allowed learning rate
        max_rate: f64,
        // Number of steps in one cycle
        step_size: usize,
    },
}

impl Scheduler {
    pub fn eta(&self, learning_rate: f64, step: usize) -> f64 {
        match self {
            Scheduler::None => learning_rate,
            Scheduler::LinearDecay { rate, step_size } => {
                learning_rate / (rate * (1 + step / step_size) as f64)
            }
            Scheduler::ExponentialDecay { rate, step_size } => {
                learning_rate * rate.powi((step / step_size) as i32)
            }
            Scheduler::OneCycle {
                max_rate,
                step_size,
            } => {
                let steps = *step_size as f64;
                let step = step % (2 * step_size);
                if step < *step_size {
                    learning_rate + (max_rate - learning_rate) * (step as f64) / (steps)
                } else {
                    max_rate - (max_rate - learning_rate) * ((step - step_size) as f64) / (steps)
                }
            }
        }
    }
}
