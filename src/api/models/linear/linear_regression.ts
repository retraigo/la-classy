import { Matrix } from "../../../helpers.ts";
import { linear } from "../../ffi/ffi.ts";
import {
  LearningRateScheduler,
  LossFunction,
  Model,
  Optimizer,
} from "../../types.ts";
import { LinearModelConfig } from "./base.ts";
import { LinearModel } from "./base.ts";

type LinearRegressionSolver = "ols" | "gd";

type LinearRegressorConfig = LinearModelConfig & {
  /**
   * How to fit the model
   * ols: Ordinary Least Squares
   * gd: Gradient Descent
   */
  solver: LinearRegressionSolver;
};
/**
 * Logistic Regression
 */
export class LinearRegressor extends LinearModel
  implements LinearRegressorConfig {
  solver: LinearRegressionSolver;
  constructor(
    { epochs, silent, learningRate, solver, optimizer, scheduler, c }: Partial<
      LinearRegressorConfig
    > = {},
  ) {
    super({ epochs, silent, learningRate, optimizer, scheduler, c });
    this.solver = solver || "ols";
  }
  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    if (this.weights === null) throw new Error("Model not trained yet.");
    const dx = new Float64Array(x.length);
    dx.set(x, 0);
    const xMatrix = new Matrix(dx, [1, x.length]);
    return xMatrix.dot(this.weights) + this.intercept;
  }
  /** Train the regressor and compute weights */
  train(x: Matrix<Float32Array> | Matrix<Float64Array>, y: ArrayLike<number>) {
    if (this.weights !== null) throw new Error("Model already trained.");
    if (!x.nRows || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.nRows}, ${y.length}).`,
      );
    }
    this.weights = new Matrix(Float64Array, [1, x.nCols]);

    const dx = new Float64Array(x.nRows * x.nCols);
    dx.set(x.data);
    const dy = Float64Array.from(y);
    const optimizerOptions = new Float64Array(
      this.optimizer.type === Optimizer.Adam
        ? 3
        : this.optimizer.type === Optimizer.MinibatchSGD
        ? 1
        : 0,
    );
    if (this.optimizer.type === Optimizer.Adam) {
      optimizerOptions[0] = this.optimizer.config.beta1 || 0.9;
      optimizerOptions[1] = this.optimizer.config.beta2 || 0.99;
      optimizerOptions[2] = this.optimizer.config.epsilon || 1e-15;
    } else if (this.optimizer.type === Optimizer.MinibatchSGD) {
      optimizerOptions[0] = this.optimizer.config.n_batches;
    }

    const schedulerOptions = new Float64Array(
      this.scheduler.type === LearningRateScheduler.OneCycleScheduler
        ? 3
        : this.scheduler.type === LearningRateScheduler.AnnealingScheduler
        ? 2
        : this.scheduler.type === LearningRateScheduler.DecayScheduler
        ? 1
        : 0,
    );
    if (this.scheduler.type === LearningRateScheduler.OneCycleScheduler) {
      schedulerOptions[0] = this.scheduler.config.initial_lr ||
        this.learningRate;
      schedulerOptions[1] = this.scheduler.config.max_lr || 0.1;
      schedulerOptions[2] = this.scheduler.config.cycle_steps || 100;
    } else if (
      this.scheduler.type === LearningRateScheduler.AnnealingScheduler
    ) {
      schedulerOptions[0] = this.scheduler.config.rate;
      schedulerOptions[1] = this.scheduler.config.step_size;
    } else if (this.scheduler.type === LearningRateScheduler.DecayScheduler) {
      schedulerOptions[0] = this.scheduler.config.rate;
    }
    this.intercept = this.solver === "ols"
      ? linear.ordinaryLeastSquares(
        new Uint8Array(this.weights.data.buffer),
        new Uint8Array(dx.buffer),
        new Uint8Array(dy.buffer),
        x.nRows,
        y.length,
        x.nCols,
        Number(this.silent),
      )
      : linear.gradientDescent(
        new Uint8Array(this.weights.data.buffer),
        new Uint8Array(dx.buffer),
        new Uint8Array(dy.buffer),
        x.nRows,
        x.nCols,
        LossFunction.MSE,
        Model.None,
        this.c,
        this.optimizer.type,
        new Uint8Array(optimizerOptions.buffer),
        this.scheduler.type,
        new Uint8Array(schedulerOptions.buffer),
        1,
        this.learningRate,
        this.epochs,
        Number(this.silent),
      );
  }
}
