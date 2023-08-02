use crate::types::Scheduler;

pub fn get_learning_rate(scheduler: &Scheduler, current: f64, step: usize, initial_lr: f64) -> f64 {
    match scheduler {
        Scheduler::None => current,
        Scheduler::ExponentialAnnealer { rate } => current * rate.powi(step as i32),
        Scheduler::LinearAnnealer { rate } => initial_lr - rate * step as f64,
        Scheduler::DecayScheduler { rate, step_size } => {
            initial_lr * rate.powi((step as i32) / (*step_size as i32)).max(1.0)
        }
        Scheduler::OneCycleScheduler {
            max_lr,
            cycle_steps,
        } => {
            let steps = *cycle_steps as f64;
            let step = step % (2 * cycle_steps);
            if step < *cycle_steps {
                initial_lr + (max_lr - initial_lr) * (step as f64) / (steps)
            } else {
                max_lr - (max_lr - initial_lr) * ((step - cycle_steps) as f64) / (steps)
            }
        }
    }
}
