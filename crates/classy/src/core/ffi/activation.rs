use crate::core::activation::Activation;

#[no_mangle]
pub unsafe extern "C" fn sigmoid_activation() -> isize {
    std::mem::transmute::<Box<Activation>, isize>(std::boxed::Box::new(Activation::Sigmoid))
}

#[no_mangle]
pub unsafe extern "C" fn no_activation() -> isize {
    std::mem::transmute::<Box<Activation>, isize>(std::boxed::Box::new(Activation::None))
}
