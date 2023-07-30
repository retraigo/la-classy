#[repr(C)]
pub enum LearningRateSchedulerFFI {
    None = 0,
    DecayScheduler = 1,
    AnnealingScheduler = 2,
    OneCycleScheduler = 3,
}
#[derive(Debug)]
pub enum LearningRateScheduler {
    None,
    DecayScheduler {
        rate: f64,
    },
    AnnealingScheduler {
        rate: f64,
        step_size: usize,
    },
    OneCycleScheduler {
        initial_lr: f64,
        max_lr: f64,
        cycle_steps: usize,
    },
}
pub fn get_learning_rate(scheduler: &LearningRateScheduler, current: f64, step: usize) -> f64 {
    match scheduler {
        LearningRateScheduler::None => current,
        LearningRateScheduler::DecayScheduler { rate } => current * rate.powi(step as i32),
        LearningRateScheduler::AnnealingScheduler { rate, step_size } => {
            if step % step_size == 0 {
                current * rate
            } else {
                current
            }
        }
        LearningRateScheduler::OneCycleScheduler {
            initial_lr,
            max_lr,
            cycle_steps,
        } => {
            let steps = *cycle_steps as f64;
            let step = step % (2 * cycle_steps);
            if step < *cycle_steps {
                initial_lr + (max_lr - initial_lr) * (step as f64) / (steps)
            } else {
                max_lr
                    - (max_lr - initial_lr) * ((step - cycle_steps) as f64) / (steps)
            }
        }
    }
}
