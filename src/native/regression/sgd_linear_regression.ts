import { sgd_linear_regression } from "../ffi/ffi.ts";

interface SgdLinearRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
  batchSize?: number;
}
/**
 * Linear Regression using Gradient Descent
 */
export class SgdLinearRegressor {
  #backend: null | Deno.PointerValue;
  epochs: number;
  silent: boolean;
  learningRate: number;
  batchSize: number;
  constructor({ epochs, silent, learningRate, batchSize }: SgdLinearRegressorConfig) {
    this.#backend = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.batchSize = batchSize || -1
  }
  destroy() {
    sgd_linear_regression.free_res(this.#backend);
    this.#backend = null;
  }
  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    if (this.#backend === null) throw new Error("Model not trained yet.");
    const dx = new Float64Array(x.length);
    dx.set(x, 0);
    return sgd_linear_regression.predict(this.#backend, dx);
  }
  /** Train the regressor */
  train(x: ArrayLike<ArrayLike<number>>, y: ArrayLike<number>) {
    if (this.#backend !== null) throw new Error("Model already trained.");
    if (!x.length || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }

    const dx = new Float64Array(x.length * x[0].length);
    for (const i in x) {
      dx.set(x[i], Number(i) * x[i].length);
    }
    const dy = Float64Array.from(y);
    const ddx = new Uint8Array(dx.buffer);
    const ddy = new Uint8Array(dy.buffer)
    this.#backend = sgd_linear_regression.train(
      ddx,
      ddy,
      x.length,
      y.length,
      x[0].length,
      this.learningRate,
      this.batchSize === -1 ? ~~(x.length / 10) : this.batchSize,
      this.epochs,
      Number(this.silent),
    );
  }
}
