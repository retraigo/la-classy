import { Matrix } from "../../../../deps.ts";
import symbols from "../../ffi/ffi.ts";
import { MaybeMatrix } from "../../types.ts";
import { linearActivation } from "../activation.ts";
import { mse } from "../loss.ts";
import { noOptimizer } from "../optimizer.ts";
import { regularizer } from "../regularizer.ts";
import { noDecay } from "../scheduler.ts";

/** Config for Gradient Descent & SAG */
export type GradientDescentConfig = {
  scheduler: Deno.PointerValue;
  optimizer: Deno.PointerValue;
  activation: Deno.PointerValue;
  loss: Deno.PointerValue;
};

export type GradientDescentTrainingConfig = {
  /** Maximum number of iterations to run. */
  epochs: number;
  /** Learning rate. Set to a small number, eg. 0.01 */
  learning_rate: number;
  /** Whether to incorporate an intercept/bias term. */
  fit_intercept: boolean;
  /** Number of minibatches to run. Set to 0 for SGD. */
  n_batches?: number;
  /** Whether to print training information each epoch. */
  silent: boolean;
  /** Minimum threshold for weight updates in each epoch. */
  tolerance: number;
  /** Number of disappointing iterations to allow before early stopping */
  patience: number;
  regularizer: Deno.PointerValue;
};

/**
 * General solver for gradient descent.
 * Uses stochastic gradient descent if
 * n_batches is greater than 1.
 */
export class GradientDescentSolver {
  #backend: Deno.PointerValue;
  weights: MaybeMatrix | null;
  bias: number;
  fit_intercept: boolean;
  constructor(data: Partial<GradientDescentConfig> = {}) {
    this.#backend = symbols.gd_solver(
      data.scheduler || noDecay(),
      data.optimizer || noOptimizer(),
      data.activation || linearActivation(),
      data.loss || mse()
    );
    this.weights = null;
    this.bias = 0;
    this.fit_intercept = false;
  }
  /**
   * Train using gradient descent.
   * @param data
   * @param targets
   * @param config
   *
   * @example
   * ```ts
   * solver.train(x_train, y_train, {
   *    learning_rate: 0.01,
   *    epochs: 1000,
   *    silent: false,
   *    n_batches: 20,
   *    fit_intercept: true,
   *    patience: 12,
   *    tolerance: 1e-6
   *    regularizer: regularizer(10, 1) // strength of 0.1 with pure l1
   * });
   * ```
   */
  train(
    data: MaybeMatrix,
    targets: MaybeMatrix,
    config: Partial<GradientDescentTrainingConfig>
  ) {
    const x = new Uint8Array(data.data.buffer);
    const y = new Uint8Array(targets.data.buffer);
    this.fit_intercept = config.fit_intercept ?? false;
    const weights = new Matrix("f64", [
      data.shape[1] + (this.fit_intercept ? 1 : 0),
      targets.shape[1],
    ]);
    const w = new Uint8Array(weights.data.buffer);
    symbols.solve(
      w,
      x,
      y,
      data.shape[0],
      data.shape[1],
      targets.shape[1],
      config.epochs || 100,
      config.learning_rate || 0.01,
      config.fit_intercept ?? false,
      config.n_batches ?? Math.floor(Math.sqrt(data.shape[0])),
      config.silent ?? true,
      config.tolerance || -1,
      config.patience || -1,
      config.regularizer || regularizer(0, 0),
      this.#backend
    );
    this.weights = weights;
  }
  /**
   * Predict the target variables using input
   * @param data
   * @returns A matrix of shape (n_samples, n_targets)
   * @example
   * ```ts
   * const res = solver.predict(x_test);
   * ```
   */
  predict(data: MaybeMatrix): Matrix<"f64"> {
    if (!this.weights) throw new Error("Solver not trained yet.");
    const x = new Uint8Array(data.data.buffer);
    const res = new Matrix("f64", [data.shape[0], this.weights.shape[1]]);
    const r = new Uint8Array(res.data.buffer);
    const w = new Uint8Array(this.weights.data.buffer);
    symbols.predict(
      r,
      w,
      x,
      data.shape[0],
      data.shape[1],
      this.weights.shape[1],
      this.fit_intercept,
      this.#backend
    );
    return res;
  }
}
