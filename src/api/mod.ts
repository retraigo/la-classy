import { Matrix } from "../../deps.ts";
import { linear } from "./ffi/ffi.ts";
import { ModelConfig, Solver } from "./types.ts";
export * from "./types.ts";
export function solve<T extends Float32Array | Float64Array>(
  config: ModelConfig,
  solver: Solver,
  x: Matrix<T>,
  y: Matrix<T>,
): [Matrix<Float64Array>, number] {
  const weights = new Float64Array(x.nCols + (config.fit_intercept ? 1 : 0));

  const configBuffer = new TextEncoder().encode(JSON.stringify(config));
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
    : [new Matrix(weights, [1, x.nCols + 1]), 0];
}
