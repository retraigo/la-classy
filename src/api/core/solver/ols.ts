import { Matrix } from "../../../../deps.ts";
import symbols from "../../ffi/ffi.ts";
import { MaybeMatrix } from "../../types.ts";

type TrainingConfig = {
  /** Whether to incorporate an intercept/bias term. */
  fit_intercept: boolean;
  /** Whether to print training information each epoch. */
  silent: boolean;
};

/**
 * A very ordinary Least Squares solver.
 */
export class OLSSolver {
  #backend: Deno.PointerValue;
  weights: MaybeMatrix | null;
  fit_intercept: boolean;
  constructor() {
    this.#backend = symbols.ols_solver();
    this.weights = null;
    this.fit_intercept = false;
  }
  train(data: MaybeMatrix, targets: MaybeMatrix, config: TrainingConfig) {
    const x = new Uint8Array(data.data.buffer);
    const y = new Uint8Array(targets.data.buffer);

    const weights = new Matrix("f64", {
      shape: [data.shape[1] + (this.fit_intercept ? 1 : 0), targets.shape[1]],
    });
    this.fit_intercept = config.fit_intercept ?? false;

    const w = new Uint8Array(weights.data.buffer);
    symbols.solve(
      w,
      x,
      y,
      data.shape[0],
      data.shape[1],
      targets.shape[1],
      1,
      1,
      config.fit_intercept ?? false,
      1,
      config.silent ?? false,
      -1,
      -1,
      null,
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
    const res = new Matrix("f64", {
      shape: [data.shape[0], this.weights.shape[1]],
    });
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
