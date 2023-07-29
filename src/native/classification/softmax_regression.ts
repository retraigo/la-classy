import { useUnique } from "../../../deps.ts";
import { Matrix } from "../../helpers.ts";
import { LogisticRegressor } from "./logistic_regression.ts";

interface LogisticRegressorConfig {
  learningRate?: number;
  silent?: boolean;
  epochs?: number;
  batches?: number;
}

interface SubLogisticRegressor {
  reg: LogisticRegressor;
  class: number;
}
/**
 * Multinomial Logistic Regression
 */
export class SoftmaxRegressor {
  epochs: number;
  silent: boolean;
  learningRate: number;
  batches: number;
  pivot: number;
  classes: null | SubLogisticRegressor[];
  constructor(
    { epochs, silent, learningRate, batches }: LogisticRegressorConfig,
  ) {
    this.epochs = epochs || 10;
    this.silent = silent || false;
    this.learningRate = learningRate || 0.01;
    this.batches = batches || 1;
    this.pivot = 0;
    this.classes = null;
  }

  /** Predict the class of an array of features */
  predict(x: ArrayLike<number>): number {
    const res = this.probs(x).sort((a, b) => b.prob - a.prob);
    return res[0].class;
  }
  probs(x: ArrayLike<number>): { prob: number; class: number }[] {
    if (!this.classes) throw new Error("Model not trained");
    const probs = new Array(this.classes.length + 1);
    let i = 0;
    let exp = 0;
    while (i < probs.length - 1) {
      const pred = this.classes[i].reg.probs(x)
      probs[i] = {
        prob: Math.exp(pred),
        class: this.classes[i].class,
      };
      exp += probs[i].prob;
      i += 1;
    }
    i = 0;
    const divisor = 1 / (exp + 1)
    while (i < probs.length - 1) {
      probs[i].prob *= divisor;
      i += 1;
    }
    probs[probs.length - 1] = {
      prob: divisor,
      class: this.pivot,
    };
    return probs;
  }
  /** Train the regressor and compute weights */
  train(x: Matrix<Float32Array> | Matrix<Float64Array>, y: ArrayLike<number>) {
    if (!x.nRows || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.nRows}, ${y.length}).`,
      );
    }
    const classes = useUnique(y);
    const dy = Float64Array.from(y);
    if (classes.length === 2) {
      throw new Error(
        "There are only two classes. Use LogisticRegressor instead.",
      );
    } else if (classes.length > 2) {
      const pivotIndex = Math.random() * classes.length;
      this.pivot = classes.splice(pivotIndex, 1)[0];
      let i = 0;
      this.classes = new Array(classes.length);
      while (i < classes.length) {
        const reg = new LogisticRegressor(this);
        const idx: number[] = [];
        const ddy = dy.filter((n, j) => {
          const sat = n === this.pivot || n === classes[i];
          if (sat) {
            idx.push(j);
            return true;
          }
          return false;
        }).map((
          n,
        ) => n === this.pivot ? 0 : 1);
        const ddx = new Matrix(Float64Array, [ddy.length, x.nCols]);
        let j = 0;
        while (j < idx.length) {
          ddx.setRow(j, x.row(idx[j]));
          j += 1;
        }
        reg.train(
          ddx,
          ddy,
        );
        this.classes[i] = { class: classes[i], reg };
        i += 1;
      }
      console.log(`Classes: `, this.classes)
    } else {
      throw new Error("Too few classes.");
    }
  }
}
