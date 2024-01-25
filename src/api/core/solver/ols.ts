import { Matrix } from "../../../../deps.ts";
import symbols from "../../ffi/ffi.ts";
import { MaybeMatrix } from "../../types.ts";

export class OLSSolver {
  #backend: Deno.PointerValue;
  weights: MaybeMatrix | null;
  fit_intercept: boolean;
  constructor() {
    this.#backend = symbols.ols_solver();
    this.weights = null;
    this.fit_intercept = false;
  }
  train(
    data: MaybeMatrix,
    targets: MaybeMatrix,
    fit_intercept = true,
    silent = true
  ) {
    const x = new Uint8Array(data.data.buffer);
    const y = new Uint8Array(targets.data.buffer);

    const weights = new Matrix<"f64">(Float64Array, [
      data.shape[1] + (fit_intercept ? 1 : 0),
      targets.shape[1],
    ]);
    this.fit_intercept = fit_intercept ?? false;

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
      fit_intercept ?? false,
      1,
      silent,
      null,
      this.#backend
    );
    this.weights = weights;
  }
  predict(data: MaybeMatrix): Matrix<"f64"> {
    if (!this.weights) throw new Error("Solver not trained yet.");
    const x = new Uint8Array(data.data.buffer);
    const res = new Matrix<"f64">(Float64Array, [
      data.shape[0],
      this.weights.shape[1],
    ]);
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
