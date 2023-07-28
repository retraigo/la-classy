import { useUnique } from "../../../deps.ts";
import { Matrix } from "../../helpers.ts";
import { ConfusionMatrix } from "../../helpers/metrics.ts";
import { logistic_regression } from "../ffi/ffi.ts";

interface LogisticRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
  batchSize?: number;
}
/**
 * Logistic Regression
 */
export class LogisticRegressor {
  #backend: null | Deno.PointerValue;
  epochs: number;
  silent: boolean;
  learningRate: number;
  batchSize: number;
  constructor({ epochs, silent, learningRate, batchSize }: LogisticRegressorConfig) {
    this.#backend = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
    this.batchSize = batchSize || 1;
  }
  confusionMatrix(
    x: Matrix<Float32Array> | Matrix<Float64Array>,
    y: ArrayLike<number>,
  ): ConfusionMatrix {
    if (this.#backend == null) throw new Error("Model not trained.");
    if (!x.nRows || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.nRows}, ${y.length}).`,
      );
    }

    const dx = new Float64Array(x.nRows * x.nCols);
    dx.set(x.data)
    const dy = Float64Array.from(y);
    const ddy = new Uint8Array(dy.buffer)
    const ddx = new Uint8Array(dx.buffer);
    const res = new Float64Array(4);
    const resPtr = new Uint8Array(res.buffer);
    const classes = useUnique(y);
    if (classes.length === 2) {
      logistic_regression.confusion_matrix(
        this.#backend,
        ddx,
        ddy,
        x.nRows,
        y.length,
        resPtr,
      );
      return new ConfusionMatrix([res[0], res[1], res[2], res[3]]);
    } else if (classes.length > 2) {
      throw new Error("Cannot classify more than two classes yet.");
    } else {
      throw new Error("Too few classes.");
    }
  }
  destroy() {
    logistic_regression.free_res(this.#backend);
    this.#backend = null;
  }
  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    if (this.#backend === null) throw new Error("Model not trained yet.");
    const dx = new Float64Array(x.length);
    dx.set(x, 0);
    return logistic_regression.predict(this.#backend, dx) > 0.5 ? 1 : 0;
  }
  /** Train the regressor */
  train(x: Matrix<Float32Array> | Matrix<Float64Array>, y: ArrayLike<number>) {
    if (this.#backend !== null) throw new Error("Model already trained.");
    if (!x.nRows || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.nRows}, ${y.length}).`,
      );
    }

    const dx = new Float64Array(x.nRows * x.nCols);
    dx.set(x.data)
    const dy = Float64Array.from(y);
    const ddy = new Uint8Array(dy.buffer)
    const ddx = new Uint8Array(dx.buffer);
    const classes = useUnique(y);
    if (classes.length === 2) {
      this.#backend = logistic_regression.train(
        ddx,
        ddy,
        x.nRows,
        y.length,
        x.nCols,
        this.learningRate,
        this.batchSize,
        this.epochs,
        Number(this.silent),
      );
    } else if (classes.length > 2) {
      throw new Error("Cannot classify more than two classes yet.");
    } else {
      throw new Error("Too few classes.");
    }
  }
}
