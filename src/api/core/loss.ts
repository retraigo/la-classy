import symbols from "../ffi/ffi.ts";

/**
 * Logistic Loss function used for binary classification.
 * It is typically used with sigmoid activation.
 */
export function binCrossEntropy(): Deno.PointerValue {
  return symbols.logit_loss();
}
/**
 * Use Cross Entropy Loss for calculating gradients.
 * It is the standard loss function for classification problems.
 * Used with softmax activation.
 */
export function crossEntropy(): Deno.PointerValue {
  return symbols.crossentropy_loss();
}
/**
 * Hinge loss is meant for binary classification using SVM.
 * Use this loss function with linear activation.
 */
export function hinge(): Deno.PointerValue {
  return symbols.hinge_loss();
}
/**
 * Use Mean Absolute Error for calculating gradients.
 * Meant for regression tasks.
 */
export function mae(): Deno.PointerValue {
  return symbols.mae_loss();
}
/**
 * Use Mean Squared Error for calculating gradients.
 * Meant for regression tasks.
 */
export function mse(): Deno.PointerValue {
  return symbols.mse_loss();
}
