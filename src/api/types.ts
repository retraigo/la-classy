import type { LearningRateScheduler } from "../../deps.ts";

export enum LossFunction {
  BinCrossEntropy = "bincrossentropy",
  CrossEntropy = "crossentropy",
  MSE = "mse",
  MAE = "mae",
}

export enum Model {
  None = "none",
  Logit = "logit",
}

// Optimizers

export enum Optimizer {
  Adam = "adam",
  None = "none"
}

export type ModelConfig = {
  epochs: number;
  silent: boolean;
  learning_rate: number;
  fit_intercept: boolean;
  model: Model;
  loss: LossFunction;
  optimizer: OptimizerConfig;
  scheduler: LearningRateScheduler;
  n_batches: number;
  c: number;
};

export interface AdamOptimizerConfig {
  beta1: number;
  beta2: number;
  epsilon: number;
}
export interface MinibatchSGDConfig {
  n_batches: number;
}

export type OptimizerConfig = { type: Optimizer.Adam; config: AdamOptimizerConfig } | { type: Optimizer.None }

export enum Solver {
  OLS = 0,
  GD = 1,
  SGD = 2,
  Minibatch = 3,
}