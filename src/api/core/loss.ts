import symbols from "../ffi/ffi.ts";

/**
 * Use Binary Cross Entropy or Logistic Loss for calculating gradients.
 */
export function binCrossEntropy(): Deno.PointerValue {
  return symbols.logit_loss();
}
/**
 * Use Mean Absolute Error for calculating gradients.
 */
export function mae(): Deno.PointerValue {
  return symbols.mae_loss();
}
/**
 * Use Mean Squared Error for calculating gradients.
 */
export function mse(): Deno.PointerValue {
  return symbols.mse_loss();
}
