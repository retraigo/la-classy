/** Linear regression output. */
interface LinearRegression {
    /** Slope of the regression. */
    slope: number;
    /** Intercept of the regression. */
    intercept: number;
    /** Predict a result `y` given an input `x`. */
    predict: (x: number) => number;
    /** Find the input `x` given a result `y`. */
    findX: (y: number) => number;
    /** The R square score. */
    r2: number;
  }
  
  /**
   * A linear regression using the Ordinary Least Squares method.
   * Requires both input and output to be single dimensional arrays
   * of equal length.
   * @param x An array of independent values (inputs).
   * @param y An array of dependent values (outputs).
   */
  export function linearRegression(x: number[], y: number[]): LinearRegression {
    // Choose the smaller array's length
    const n = Math.min(x.length, y.length);
    // Calculate mean of both arrrays.
    const mean = [0, 0];
    let i = 0;
    while (i < n) {
      mean[0] += x[i];
      mean[1] += y[i];
      ++i;
    }
    console.log("MEANS", mean)
    mean[0] = mean[0] / n;
    mean[1] = mean[1] / n;
  
    // Calculate standard deviation to find slope and hence, intercept.
    const stddev = [0, 0];
    i = 0;
    while (i < n) {
      stddev[0] += (x[i] - mean[0]) * (y[i] - mean[1]);
      stddev[1] += (x[i] - mean[0]) * (x[i] - mean[0]);
      ++i;
    }
    console.log(stddev, mean)
    const slope = stddev[0] / stddev[1];
    const intercept = mean[1] - slope * mean[0];
    const predict = (x: number) => intercept + (slope * x);
    const findX = (y: number) => (y - intercept) / slope;
  
    // Calculate SSR and SST to find R2
    let sse = 0, sst = 0;
    i = 0;
    while (i < n) {
      sse += Math.pow(y[i] - predict(x[i]), 2);
      sst += Math.pow(y[i] - mean[1], 2);
      ++i;
    }
    return {
      slope: slope,
      intercept: intercept,
      predict,
      findX,
      r2: 1 - sse / sst,
    };
  }