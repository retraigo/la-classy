import symbols from "../ffi/ffi.ts";

/**
 * Use sigmoid activation function. It squashes 
 * values between the range [0, 1] and is typically
 * used for Logistic Regression.
 */
export function sigmoidActivation(): Deno.PointerValue {
  return symbols.sigmoid_activation();
}
/**
 * Use no activation function. Usually used for regression
 * tasks or Linear SVM.
 */
export function linearActivation(): Deno.PointerValue {
  return symbols.no_activation();
}
/**
 * Use the hyperbolic tangent activation function. It
 * squashes values between the range [-1, 1].
 */
export function tanhActivation(): Deno.PointerValue {
  return symbols.no_activation();
}

/**
 * Use the softmax activation function. 
 * It is used for multi-class classification
 * problems when the targets are categorically encoded.
 */
export function softmaxActivation(): Deno.PointerValue {
  return symbols.softmax_activation();
}
