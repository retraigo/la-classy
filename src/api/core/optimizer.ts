import symbols from "../ffi/ffi.ts";

/**
 * ADAM optimizer uses moments to adaptively adjust the learning rates for each parameter.
 * @param inputSize Size of each individual data point (number of columns in data).
 * @param outputSize Size of output (number of columns in target).
 * @param beta1 Hyperparameter for ADAM
 * @param beta2 Hyperparameter for ADAM
 * @param epsilon Set to a very small value like 1e-14
 */
export function adamOptimizer(
  inputSize: number,
  outputSize: number,
  beta1 = 0.9,
  beta2 = 0.99,
  epsilon = 1e-14,
): Deno.PointerValue {
  return symbols.adam_optimizer(beta1, beta2, epsilon, inputSize, outputSize);
}

/**
 * RMSProp optimizer maintains an exponentially decaying average of 
 * squared gradients for each parameter.
 * @param inputSize Size of each individual data point (number of columns in data).
 * @param outputSize Size of output (number of columns in target).
 * @param beta1 Hyperparameter for ADAM
 * @param beta2 Hyperparameter for ADAM
 * @param epsilon Set to a very small value like 1e-14
 */
export function rmsPropOptimizer(
  inputSize: number,
  outputSize: number,
  decayRate = 0.9,
  epsilon = 1e-14,
): Deno.PointerValue {
  return symbols.rmsprop_optimizer(decayRate, epsilon, inputSize, outputSize);
}

/**
 * Use no optimizer
 */
export function noOptimizer(): Deno.PointerValue {
  return symbols.no_optimizer();
}

