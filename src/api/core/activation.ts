import symbols from "../ffi/ffi.ts";

/**
 * Use sigmoid activation function.
 */
export function sigmoidActivation(): Deno.PointerValue {
  return symbols.sigmoid_activation();
}
/**
 * Use no activation function.
 */
export function linearActivation(): Deno.PointerValue {
  return symbols.no_activation();
}
/**
 * Use tanh activation function.
 */
export function tanhActivation(): Deno.PointerValue {
  return symbols.no_activation();
}
