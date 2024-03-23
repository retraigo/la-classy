use crate::core::scheduler::Scheduler;

#[no_mangle]
pub unsafe extern "C" fn linear_decay_scheduler(rate: f32, step_size: usize) -> isize {
    std::mem::transmute::<Box<Scheduler>, isize>(std::boxed::Box::new(Scheduler::LinearDecay {
        rate,
        step_size,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn exponential_decay_scheduler(rate: f32, step_size: usize) -> isize {
    std::mem::transmute::<Box<Scheduler>, isize>(std::boxed::Box::new(
        Scheduler::ExponentialDecay { rate, step_size },
    ))
}

#[no_mangle]
pub unsafe extern "C" fn one_cycle_scheduler(max_rate: f32, step_size: usize) -> isize {
    std::mem::transmute::<Box<Scheduler>, isize>(std::boxed::Box::new(Scheduler::OneCycle {
        max_rate,
        step_size,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn no_decay_scheduler() -> isize {
    std::mem::transmute::<Box<Scheduler>, isize>(std::boxed::Box::new(Scheduler::None))
}