#[repr(C)] #[derive(Debug)]
pub enum LossFunction {
    LOGIT = 1,
    MSE = 2,
}

#[repr(C)] #[derive(Debug)]
pub enum Convertor {
    NONE = 0,
    LOGIT = 1,
}
