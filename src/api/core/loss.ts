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
 * Avoid using the `tolerance` hyperparameter for training
 * when using hinge loss.
 */
export function hinge(): Deno.PointerValue {
  return symbols.hinge_loss();
}
/**
 * Huber loss is a loss function for regression and is less
 * sensitive to outliers than the squared error loss
 */
export function huber(delta = 1.5): Deno.PointerValue {
  return symbols.huber_loss(delta);
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
/**
 * A variant of the Huber loss function used for
 * binary classification. It is a smoothed version
 * of hinge loss and is more robust to outliers.
 */
export function smoothedHinge(): Deno.PointerValue {
  return symbols.smooth_hinge_loss();
}
/**
 * A robust loss function for regression problems.
 * @param c Tuning parameter for Tukey's Biweight
 */
export function tukey(c = 4.685): Deno.PointerValue {
  return symbols.tukey_loss(c);
}
