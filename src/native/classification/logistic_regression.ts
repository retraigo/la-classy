import { useUnique } from "../../../deps.ts";
import { logistic_regression } from "../ffi/ffi.ts";

interface LogisticRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
}
class ConfusionMatrix {
  truePositive: number;
  falsePositive: number;
  trueNegative: number;
  falseNegative: number;
  true: number;
  false: number;
  size: number;
  constructor([tp, fn, fp, tn]: [number, number, number, number]) {
    this.truePositive = tp;
    this.falseNegative = fn;
    this.falsePositive = fp;
    this.trueNegative = tn;
    this.true = tn + tp;
    this.false = fn + fp;
    this.size = this.true + this.false;
  }
  valueOf(): [number, number, number, number] {
    return [
      this.truePositive,
      this.falseNegative,
      this.falsePositive,
      this.trueNegative,
    ];
  }
  [Symbol.for("Deno.customInspect")]() {
    return `\n${this.truePositive}\t${this.falseNegative}\n${this.falsePositive}\t${this.trueNegative}`;
  }
}

/**
 * Logistic Regression
 */
export class LogisticRegressor {
  #backend: null | Deno.PointerValue;
  epochs: number;
  silent: boolean;
  learningRate: number;
  constructor({ epochs, silent, learningRate }: LogisticRegressorConfig) {
    this.#backend = null;
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.001;
  }
  confusionMatrix(
    x: ArrayLike<ArrayLike<number>>,
    y: ArrayLike<number>,
  ): ConfusionMatrix {
    if (this.#backend == null) throw new Error("Model not trained.");
    if (!x.length || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }

    const dx = new Float64Array(x.length * x[0].length);
    for (const i in x) {
      dx.set(x[i], Number(i) * x[i].length);
    }
    const dy = Uint8Array.from(y);
    const ddx = new Uint8Array(dx.buffer);
    const res = new Float64Array(4);
    const resPtr = new Uint8Array(res.buffer);
    const classes = useUnique(y);
    if (classes.length === 2) {
      logistic_regression.confusion_matrix(
        this.#backend,
        ddx,
        dy,
        x.length,
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
    const dy = Uint8Array.from(y);
    const ddx = new Uint8Array(dx.buffer);
    const classes = useUnique(y);
    if (classes.length === 2) {
      this.#backend = logistic_regression.train(
        ddx,
        dy,
        x.length,
        y.length,
        x[0].length,
        this.learningRate,
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
