#[repr(C)]
#[derive(Debug)]
pub enum LossFunction {
    BinCrossEntropy = 1,
    MSE = 2,
}

#[repr(C)]
#[derive(Debug)]
pub enum Convertor {
    None = 0,
    Logit = 1,
}
