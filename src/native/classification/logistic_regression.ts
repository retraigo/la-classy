import { useUnique } from "../../../deps.ts";
import { Matrix } from "../../helpers.ts";
import { sigmoid } from "../../helpers/functions.ts";
import { ConfusionMatrix } from "../../helpers/metrics.ts";
import { linear } from "../ffi/ffi.ts";

interface LogisticRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
  batches?: number;
}
/**
 * Logistic Regression
 */
export class LogisticRegressor {
  weights: Matrix<Float64Array> | null;
  epochs: number;
  silent: boolean;
  learningRate: number;
  batches: number;
  intercept: number;
  constructor(
    { epochs, silent, learningRate, batches }: LogisticRegressorConfig,
  ) {
    this.weights = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.batches = batches || 1;
    this.intercept = 0;
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
      this.intercept = linear.gradientDescent(
        new Uint8Array(this.weights.data.buffer),
        new Uint8Array(dx.buffer),
        new Uint8Array(dy.buffer),
        x.nRows,
        y.length,
        x.nCols,
        1,
        1,
        0,
        this.learningRate,
        this.batches,
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
