import symbols from "../ffi/ffi.ts";

/**
 * Initialize an ADAM optimizer
 * @param beta1 Hyperparameter for ADAM
 * @param beta2 Hyperparameter for ADAM
 * @param epsilon Set to a very small value like 1e14
 * @param inputSize Size of each individual data point (number of columns in data).
 */
export function adamOptimizer(
  beta1: number,
  beta2: number,
  epsilon: number,
  inputSize: number,
): Deno.PointerValue {
  return symbols.adam_optimizer(beta1, beta2, epsilon, inputSize);
}
/**
 * Use no optimizer
 */
export function noOptimizer(): Deno.PointerValue {
  return symbols.no_optimizer();
}

