import { Matrix } from "../../../../deps.ts";
import symbols from "../../ffi/ffi.ts";
import { linearActivation } from "../activation.ts";
import { mse } from "../loss.ts";
import { noOptimizer } from "../optimizer.ts";
import { regularizer } from "../regularizer.ts";
import { noDecay } from "../scheduler.ts";
import { GradientDescentConfig } from "./gradient_descent.ts";


type TrainingConfig = {
  epochs: number;
  learning_rate: number;
  fit_intercept: boolean;
  silent: boolean;
  regularizer: Deno.PointerValue;
};

export class SGDSolver {
  #backend: Deno.PointerValue;
  weights: Matrix<"f64"> | null;
  bias: number;
  constructor(data: Partial<GradientDescentConfig> = {}) {
    this.#backend = symbols.sgd_solver(
      data.scheduler || noDecay(),
      data.optimizer || noOptimizer(),
      data.activation || linearActivation(),
      data.loss || mse(),
    );
    this.weights = null;
    this.bias = 0;
  }
  train(
    data: Matrix<"f64">,
    targets: Matrix<"f64">,
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
      config.fit_intercept ?? false,
      0,
      config.silent ?? true,
      config.regularizer || regularizer(0, 0),
      this.#backend,
    );
    if (config.fit_intercept) {
      this.weights = new Matrix(weights.slice(1), [1]);
      this.bias = weights[0];
    } else {
      this.weights = new Matrix(weights, [1]);
    }
  }
}
