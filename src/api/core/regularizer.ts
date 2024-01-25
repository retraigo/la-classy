import symbols from "../ffi/ffi.ts";

/**
 * Initialize a regularizer.
 * @param c Inverse of the regularization strength. 
 * Smaller value indicates stronger regularization.
 * @param l1Ratio Ratio between L1 and L2 strength. 
 * Value of 0 will apply pure L1 (lasso) regularization
 * and 1 will apply pure L2 (ridge) regularization.
 * @returns Regularizer for use in solver.
 * 
 * @example
 * ```ts
 * solver.train(x_train, y_train, {
 *    learning_rate: 0.01,
 *    epochs: 1000,
 *    silent: false,
 *    n_batches: 20,
 *    patience: 12,
 *    regularizer: regularizer(10, 1) // strength of 0.1 with pure l1
 * });
 * ```
 */
export function regularizer(
  c: number,
  l1Ratio: number,
): Deno.PointerValue {
  return symbols.regularizer(c, l1Ratio);
}
