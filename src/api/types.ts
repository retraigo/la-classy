export enum LossFunction {
  BinCrossEntropy = 1,
  MSE = 2,
}

export enum Model {
  None = 0,
  Logit = 1,
}

// Optimizers

export enum Optimizer {
  Adam = 1,
  SGD = 2,
  MinibatchSGD = 3,
  GD = 4,
}

export interface AdamOptimizerConfig {
  beta1: number;
  beta2: number;
  epsilon: number;
}
export interface MinibatchSGDConfig {
  n_batches: number;
}

export type OptimizerConfig =
  | { type: Optimizer.Adam; config: AdamOptimizerConfig }
  | { type: Optimizer.SGD }
  | { type: Optimizer.MinibatchSGD; config: MinibatchSGDConfig }
  | { type: Optimizer.GD };

// Learning Rate Schedulers

export enum LearningRateScheduler {
  None = 0,
  DecayScheduler = 1,
  AnnealingScheduler = 2,
  OneCycleScheduler = 3,
}
export type LearningRateSchedulerConfig =
  | {
    type: LearningRateScheduler.OneCycleScheduler;
    config: { initial_lr: number; max_lr: number; cycle_steps: number };
  }
  | {
    type: LearningRateScheduler.AnnealingScheduler;
    config: { rate: number; step_size: number };
  }
  | { type: LearningRateScheduler.DecayScheduler; config: { rate: number } }
  | { type: LearningRateScheduler.None };
