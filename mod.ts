/**
 * Single Layer Perceptron (SLP) library for Deno.
 * - Uses FFI (requires `--unstable-ffi`)
 * - Does not support GPU
 * 
 * @example
 * ```ts
 * import { Matrix } from "jsr:@lala/appraisal@0.7.3";
 * import { GradientDescentSolver, adamOptimizer, huber } from "jsr:@lala/appraisal@1.2.0";
 * 
 * const x = [100, 23, 53, 56, 12, 98, 75];
 * const y = x.map((a) => [a * 6 + 13, a * 4 + 2]);
 * 
 * const solver = new GradientDescentSolver({
 *     // Huber loss is a mix of MSE and MAE
 *     loss: huber(),
 *     // ADAM optimizer with 1 + 1 input for intercept, 2 outputs.
 *     optimizer: adamOptimizer(2, 2)
 * });
 * 
 * // Train for 700 epochs in 2 minibatches
 * solver.train(
 *     new Matrix(x.map(n => [n]), undefined, "f64"),
 *     new Matrix(y, undefined, "f64"),
 *     { silent: false, fit_intercept: true, epochs: 700, n_batches: 2 },
 * );
 * 
 * const res = solver.predict(
 *     new Matrix(x.map(n => [n]), undefined, "f64"),
 * );
 * 
 * for (let i = 0; i < res.nRows; i += 1) {
 *     console.log(Array.from(res.row(i)), y[i]);
 * }
 * ```
 * @module
 */

export * from "./src/mod.ts"