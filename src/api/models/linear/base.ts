import { Matrix } from "../../../helpers.ts";
import {
  LearningRateScheduler,
  LearningRateSchedulerConfig,
  Optimizer,
  OptimizerConfig,
} from "../../types.ts";

export interface LinearModelConfig {
  /** Learning rate if solver is set to "gd". Set it to a small value. */
  learningRate: number;
  /** Whether to output logs while training */
  silent: boolean;
  /** Number of epochs to train for if solver is set to "gd" */
  epochs: number;
  /** Optimizer for gradient descent */
  optimizer: OptimizerConfig;
  /** Scheduler for updating learning rate */
  scheduler: LearningRateSchedulerConfig;
  /** L1 Regularization Parameter */
  c: number;
}
/**
 * Base Linear Model
 */
export class LinearModel implements LinearModelConfig {
  weights: Matrix<Float64Array> | null;
  epochs: number;
  silent: boolean;
  learningRate: number;
  intercept: number;
  scheduler: LearningRateSchedulerConfig;
  optimizer: OptimizerConfig;
  c: number;
  constructor(
    { epochs, silent, learningRate, optimizer, scheduler, c }: Partial<
      LinearModelConfig
    > = {},
  ) {
    this.weights = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.intercept = 0;
    this.optimizer = optimizer || { type: Optimizer.SGD };
    this.scheduler = scheduler || { type: LearningRateScheduler.None };
    this.c = c || 0;
  }
  /** Predict the class of an array of features */
  predict(_x: ArrayLike<number>): number {
    return 0;
  }
  /** Train the regressor and compute weights */
  train(
    _x: Matrix<Float32Array> | Matrix<Float64Array>,
    _y: ArrayLike<number>,
  ) {
    null;
  }
}
