export enum LossFunction {
    BinCrossEntropy = 1,
    MSE = 2,
}

export enum Model {
    None = 0,
    Logit = 1,
}

export enum Optimizer {
    Adam = 1,
    SGD = 2,
    MinibatchSGD =  3,
    GD = 4,
}