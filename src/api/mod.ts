import { Matrix } from "../../deps.ts";
import { Scheduler } from "../helpers.ts";
import { linear } from "./ffi/ffi.ts";
import {
  LossFunction,
  Model,
  ModelConfig,
  Optimizer,
  Solver,
} from "./types.ts";
export * from "./types.ts";
export function solve<T extends Float32Array | Float64Array>(
  config: Partial<ModelConfig>,
  solver: Solver,
  x: Matrix<T>,
  y: Matrix<T>,
): [Matrix<Float64Array>, number] {
  const weights = new Float64Array(x.nCols + (config.fit_intercept ? 1 : 0));

  const conf: ModelConfig = {
    epochs: config.epochs || 1,
    silent: config.silent ?? true,
    learning_rate: config.learning_rate || 0.01,
    fit_intercept: config.fit_intercept || false,
    model: config.model || Model.None,
    loss: config.loss || LossFunction.MSE,
    optimizer: config.optimizer || { type: Optimizer.None },
    scheduler: config.scheduler || {type: Scheduler.None},
    n_batches: solver === Solver.Minibatch ? config.n_batches ? config.n_batches : 1 : 1,
    c: config.c || 0,
  };

  const configBuffer = new TextEncoder().encode(JSON.stringify(conf));
  linear.solve(
    new Uint8Array(weights.buffer),
    new Uint8Array(x.data.buffer),
    new Uint8Array(y.data.buffer),
    x.nRows,
    x.nCols,
    configBuffer,
    configBuffer.length,
    solver,
  );
  return config.fit_intercept
    ? [new Matrix(weights.slice(0, x.nCols), [1, x.nCols]), weights[x.nCols]]
    : [new Matrix(weights, [1, x.nCols]), 0];
}
