export class LinearRegressor {
  /** Slope of the regression. */
  slope: number;
  /** Intercept of the regression. */
  intercept: number;
  /** The R square score. */
  r2: number;
  constructor() {
    this.slope = NaN;
    this.intercept = NaN;
    this.r2 = 0;
  }
  /** Find the input `x` given a result `y`. */
  findX(y: number): number {
    if (isNaN(this.slope) || isNaN(this.intercept)) {
      throw new Error("Model not trained yet.");
    }
    return (y - this.intercept) / this.slope;
  }
  /** Predict a result `y` given an input `x`. */
  predict(x: number): number {
    if (isNaN(this.slope) || isNaN(this.intercept)) {
      throw new Error("Model not trained yet.");
    }
    return this.intercept + (this.slope * x);
  }
  /**
   * A linear regression using the Ordinary Least Squares method.
   * Requires both input and output to be single dimensional arrays
   * of equal length.
   * @param x An array of independent values (inputs).
   * @param y An array of dependent values (outputs).
   */
  train(x: ArrayLike<number>, y: ArrayLike<number>): LinearRegressor {
    // Choose the smaller array's length
    const n = Math.min(x.length, y.length);
    if (n === 0) {
      throw new Error(
        `Arrays must not be empty. Received size (${x.length}, ${y.length}).`,
      );
    }
    const mean = [0, 0];
    let i = 0;
    while (i < n) {
      mean[0] += x[i];
      mean[1] += y[i];
      ++i;
    }
    mean[0] = mean[0] / n;
    mean[1] = mean[1] / n;
    console.log("mean", mean)
    const stddev = [0, 0];
    i = 0;
    while (i < n) {
      stddev[0] += (x[i] - mean[0]) * (y[i] - mean[1]);
      stddev[1] += (x[i] - mean[0]) * (x[i] - mean[0]);
      ++i;
    }
    const slope = stddev[0] / stddev[1];
    const intercept = mean[1] - slope * mean[0];
    const predict = (x: number) => intercept + (slope * x);
    const findX = (y: number) => (y - intercept) / slope;

    let sse = 0, sst = 0;
    i = 0;
    while (i < n) {
      sse += Math.pow(y[i] - predict(x[i]), 2);
      sst += Math.pow(y[i] - mean[1], 2);
      ++i;
    }
    this.slope = slope;
    this.intercept = intercept;
    this.predict = predict;
    this.findX = findX;
    this.r2 = 1 - sse / sst;
    return this;
  }
}
