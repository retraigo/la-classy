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
export function noActivation(): Deno.PointerValue {
  return symbols.no_activation();
}
