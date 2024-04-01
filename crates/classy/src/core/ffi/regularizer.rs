use crate::core::regularization::Regularization;

#[no_mangle]
pub unsafe extern "C" fn regularizer(c: f32, l1_ratio: f32) -> isize {
    let reg = Regularization::from(c, l1_ratio);
    std::mem::transmute::<Box<Regularization>, isize>(std::boxed::Box::new(reg))
}