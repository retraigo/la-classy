import { linear } from "../ffi/ffi.ts";
export class LinearRegressor {
  slope: number;
  intercept: number;
  r2: number;
  constructor() {
    this.slope = NaN;
    this.intercept = NaN;
    this.r2 = NaN;
  }
  predict(x: number): number {
    if (isNaN(this.slope) || isNaN(this.intercept)) {
      throw new Error("Model not trained yet.");
    }
    return this.intercept + (this.slope * x);
  }
  train(x: ArrayLike<number>, y: ArrayLike<number>) {
    if (!x.length || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }
    const dx = Float64Array.from(x);
    const dy = Float64Array.from(y);
    const res = new Float64Array(3);
    linear.ordinaryLeastSquares(
      new Uint8Array(dx.buffer),
      new Uint8Array(dy.buffer),
      x.length,
      y.length,
      new Uint8Array(res.buffer),
    );
    this.r2 = res[2];
    this.slope = res[0];
    this.intercept = res[1];
  }
}
