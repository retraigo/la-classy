import { Matrix } from "../../helpers.ts";
import { linear } from "../ffi/ffi.ts";

interface SgdLinearRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
  batchSize?: number;
}
/**
 * Logistic Regression
 */
export class SgdLinearRegressor {
  weights: Matrix<Float64Array> | null;
  epochs: number;
  silent: boolean;
  learningRate: number;
  batchSize: number;
  intercept: number;
  constructor(
    { epochs, silent, learningRate, batchSize }: SgdLinearRegressorConfig,
  ) {
    this.weights = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.batchSize = batchSize || -1;
    this.intercept = 0;
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
    this.intercept = linear.gradientDescent(
      new Uint8Array(this.weights.data.buffer),
      new Uint8Array(dx.buffer),
      new Uint8Array(dy.buffer),
      x.nRows,
      y.length,
      x.nCols,
      2,
      0,
      1,
      this.learningRate,
      this.batchSize === -1 ? x.nRows / 4 : this.batchSize,
      this.epochs,
      Number(this.silent),
    );
  }
}
