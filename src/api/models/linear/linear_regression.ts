import { Matrix } from "../../../helpers.ts";
import { linear } from "../../ffi/ffi.ts";
import { LossFunction, Model, Optimizer } from "../../types.ts";

type LinearRegressionSolver = "ols" | "gd";

interface LinearRegressorConfig {
  /** Learning rate if solver is set to "gd". Set it to a small value. */
  learningRate?: number;
  /** Whether to output logs while training */
  silent?: boolean;
  /** Number of epochs to train for if solver is set to "gd" */
  epochs?: number;
  /** Number of batches if optimizer is set to MinibatchSGD */
  batches?: number;
  /** Optimizer if case solver is set to "gd" */
  optimizer?: Optimizer;
  /**
   * How to fit the model
   * ols: Ordinary Least Squares
   * gd: Gradient Descent
   */
  solver?: LinearRegressionSolver;
}
/**
 * Logistic Regression
 */
export class LinearRegressor implements LinearRegressorConfig {
  weights: Matrix<Float64Array> | null;
  epochs: number;
  silent: boolean;
  learningRate: number;
  batches: number;
  intercept: number;
  optimizer: Optimizer;
  solver: LinearRegressionSolver;
  constructor(
    { epochs, silent, learningRate, batches, solver, optimizer }:
      LinearRegressorConfig = {},
  ) {
    this.weights = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.batches = batches || 1;
    this.intercept = 0;
    this.optimizer = optimizer || Optimizer.SGD;
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
        y.length,
        x.nCols,
        LossFunction.MSE, // Use MSE as loss function
        Model.None, // Use no special model
        this.optimizer,
        new Float64Array([0.99, 0.999, 1e-15]), // Pass buffer for adam options
        1,
        this.learningRate,
        this.batches,
        this.epochs,
        Number(this.silent),
      );
  }
}
