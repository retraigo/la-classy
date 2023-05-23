import { linear_regression } from "../ffi.ts";
export class LinearRegressor {
  slope: number;
  intercept: number;
  r2: number;
  constructor() {
    this.slope = NaN;
    this.intercept = NaN;
    this.r2 = NaN;  }
  predict(x: number): number {
    if (isNaN(this.slope) || isNaN(this.intercept)) throw new Error("Model not trained yet.");
    return this.intercept + (this.slope * x);
  }
  train(x: ArrayLike<number>, y: ArrayLike<number>) {
    if (!x.length || !y.length) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }
    const dx = Float32Array.from(x);
    const dy = Float32Array.from(y);
    const ddx = new Uint8Array(dx.buffer);
    const ddy = new Uint8Array(dy.buffer);
    const linregress = linear_regression.linear_regression(
      ddx,
      ddy,
      x.length,
      y.length,
    );
    this.r2 = linear_regression.get_r2(linregress);
    this.slope = linear_regression.get_slope(linregress);
    this.intercept = linear_regression.get_intercept(linregress);
    linear_regression.free_res(linregress)
  }
}
