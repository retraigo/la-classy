import { logistic_regression } from "../ffi/ffi.ts";
/**
 * Logistic Regression
 */
export class LogisticRegressor {
  #backend: null | Deno.PointerValue;
  epochs: number;
  silent: boolean;
  constructor({ epochs, silent }: { epochs: number; silent: boolean }) {
    this.#backend = null;
    this.epochs = epochs;
    this.silent = silent;
  }
  destroy() {
    logistic_regression.free_res(this.#backend)
    this.#backend = null;
  }
  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    if (this.#backend === null) throw new Error("Model not trained yet.");
    const dx = new Float32Array(x.length)
    dx.set(x, 0)
    return logistic_regression.predict(this.#backend, dx) > 0 ? 1 : 0;
  }
  /** Train the regressor */
  train(x: ArrayLike<ArrayLike<number>>, y: ArrayLike<number>) {
    if (this.#backend !== null) throw new Error("Model already trained.");
    if (!x.length || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }

    const dx = new Float32Array(x.length * x[0].length);
    for (const i in x) {
        dx.set(x[i], Number(i) * x[i].length);
    }
    const dy = Float32Array.from(y);
    const ddx = new Uint8Array(dx.buffer);
    const ddy = new Uint8Array(dy.buffer);
    this.#backend = logistic_regression.train(
      ddx,
      ddy,
      x.length,
      y.length,
      x[0].length,
      this.epochs,
      Number(this.silent),
    );
  }
}
