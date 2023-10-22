import { Matrix } from "../../../helpers.ts";
import symbols from "../../ffi/ffi.ts";
import { noActivation } from "../activation.ts";
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
  silent: boolean;
  regularizer: Deno.PointerValue;
};

export class GradientDescentSolver {
  #backend: Deno.PointerValue;
  weights: Matrix<Float64Array> | null;
  bias: number;
  constructor(data: Partial<GradientDescentConfig> = {}) {
    this.#backend = symbols.gd_solver(
      data.scheduler || noDecay(),
      data.optimizer || noOptimizer(),
      data.activation || noActivation(),
      data.loss || mse(),
    );
    this.weights = null;
    this.bias = 0;
  }
  train(
    data: Matrix<Float64Array>,
    targets: Matrix<Float64Array>,
    config: Partial<TrainingConfig>,
  ) {
    const x = new Uint8Array(data.data.buffer);
    const y = new Uint8Array(targets.data.buffer);

    const weights = new Float64Array(
      config.fit_intercept ? data.nCols + 1 : data.nCols,
    );

    const w = new Uint8Array(weights.buffer);

    symbols.solve(
      w,
      x,
      y,
      data.nRows,
      data.nCols,
      config.epochs || 100,
      config.learning_rate || 0.01,
      config.fit_intercept || false,
      0,
      config.silent ?? true,
      config.regularizer || regularizer(0, 0),
      this.#backend,
    );
    if (config.fit_intercept) {
      this.weights = new Matrix(weights.slice(0, weights.length - 1), [1]);
      this.bias = weights[weights.length - 1];
    } else {
      this.weights = new Matrix(weights, [1]);
    }
  }
}
