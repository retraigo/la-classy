import { Matrix } from "../../../../deps.ts";
import symbols from "../../ffi/ffi.ts";
import { linearActivation } from "../activation.ts";
import { mse } from "../loss.ts";
import { noOptimizer } from "../optimizer.ts";
import { regularizer } from "../regularizer.ts";
import { noDecay } from "../scheduler.ts";

export type GradientDescentConfig = {
  scheduler: Deno.PointerValue;
  optimizer: Deno.PointerValue;
  activation: Deno.PointerValue;
  loss: Deno.PointerValue;
};

type TrainingConfig = {
  epochs: number;
  learning_rate: number;
  fit_intercept: boolean;
  n_batches: number;
  silent: boolean;
  regularizer: Deno.PointerValue;
};

export class GradientDescentSolver {
  #backend: Deno.PointerValue;
  weights: Matrix<"f64"> | null;
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
  train(
    data: Matrix<"f64">,
    targets: Matrix<"f64">,
    config: Partial<TrainingConfig>
  ) {
    const x = new Uint8Array(data.data.buffer);
    const y = new Uint8Array(targets.data.buffer);

    const weights = new Matrix<"f64">(Float64Array, [
      data.nCols + (this.fit_intercept ? 1 : 0),
      targets.nCols,
    ]);
    this.fit_intercept = config.fit_intercept ?? false;

    const w = new Uint8Array(weights.data.buffer);
    symbols.solve(
      w,
      x,
      y,
      data.nRows,
      data.nCols,
      targets.nCols,
      config.epochs || 100,
      config.learning_rate || 0.01,
      config.fit_intercept ?? false,
      config.n_batches || Math.floor(Math.sqrt(data.nRows)),
      config.silent ?? true,
      config.regularizer || regularizer(0, 0),
      this.#backend
    );
    this.weights = weights;
  }
  predict(data: Matrix<"f64">): Matrix<"f64"> {
    if (!this.weights) throw new Error("Solver not trained yet.");
    const x = new Uint8Array(data.data.buffer);
    const res = new Matrix<"f64">(Float64Array, [
      data.nRows,
      this.weights.nCols,
    ]);
    const r = new Uint8Array(res.data.buffer);
    const w = new Uint8Array(this.weights.data.buffer);
    symbols.predict(
      r,
      w,
      x,
      data.nRows,
      data.nCols,
      this.weights.nCols,
      this.fit_intercept,
      this.#backend
    );
    return res;
  }
}
