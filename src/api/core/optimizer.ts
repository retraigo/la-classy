import symbols from "../ffi/ffi.ts";

/**
 * Initialize an ADAM optimizer
 * @param beta1 Hyperparameter for ADAM
 * @param beta2 Hyperparameter for ADAM
 * @param epsilon Set to a very small value like 1e-14
 * @param inputSize Size of each individual data point (number of columns in data).
 */
export function adamOptimizer(
  inputSize: number,
  beta1 = 0.9,
  beta2 = 0.99,
  epsilon = 1e-14,
): Deno.PointerValue {
  return symbols.adam_optimizer(beta1, beta2, epsilon, inputSize);
}
/**
 * Use no optimizer
 */
export function noOptimizer(): Deno.PointerValue {
  return symbols.no_optimizer();
}

