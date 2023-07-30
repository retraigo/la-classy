import { useUnique } from "../../../../deps.ts";
import { ConfusionMatrix, Matrix, sigmoid } from "../../../helpers.ts";
import { linear } from "../../ffi/ffi.ts";
import { LearningRateScheduler, LossFunction, Model, Optimizer } from "../../types.ts";
import { LinearModel, LinearModelConfig } from "./base.ts";

/**
 * Logistic Regression
 */
export class LogisticRegressor extends LinearModel {
  constructor(
    { epochs, silent, learningRate, optimizer, scheduler, c }: Partial<
      LinearModelConfig
    > = {},
  ) {
    super({ epochs, silent, learningRate, optimizer, scheduler, c });
  }
  /** Output a confusion matrix */
  confusionMatrix(
    x: Matrix<Float32Array> | Matrix<Float64Array>,
    y: ArrayLike<number>,
  ): ConfusionMatrix {
    if (this.weights == null) throw new Error("Model not trained.");
    if (!x.nRows || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.nRows}, ${y.length}).`,
      );
    }
    const res = new Uint32Array(4);
    const classes = useUnique(y);
    if (classes.length === 2) {
      let i = 0;
      while (i < x.length) {
        const yi = this.predict(x.row(i));
        if (yi === 1 && y[i] === 1) res[0] += 1;
        else if (yi === 0 && y[i] === 1) res[1] += 1;
        else if (yi === 1 && y[i] === 0) res[2] += 1;
        else res[3] += 1;
        i += 1;
      }
      return new ConfusionMatrix([res[0], res[1], res[2], res[3]]);
    } else if (classes.length > 2) {
      throw new Error("Cannot classify more than two classes yet.");
    } else {
      throw new Error("Too few classes.");
    }
  }
  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    return sigmoid(this.probs(x)) > 0.5 ? 1 : 0;
  }
  probs(x: ArrayLike<number>): number {
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
    const classes = useUnique(y);
    if (classes.length === 2) {
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
      this.intercept = linear.gradientDescent(
        new Uint8Array(this.weights.data.buffer),
        new Uint8Array(dx.buffer),
        new Uint8Array(dy.buffer),
        x.nRows,
        x.nCols,
        LossFunction.BinCrossEntropy,
        Model.Logit,
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
    } else if (classes.length > 2) {
      throw new Error("Cannot classify more than two classes yet.");
    } else {
      throw new Error("Too few classes.");
    }
  }
}
