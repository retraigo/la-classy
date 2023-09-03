use crate::core::regularization::Regularization;

#[no_mangle]
pub unsafe extern "C" fn regularizer(c: f64, l1_ratio: f64) -> isize {
    let reg = Regularization::from(c, l1_ratio);
    std::mem::transmute::<Box<Regularization>, isize>(std::boxed::Box::new(reg))
}