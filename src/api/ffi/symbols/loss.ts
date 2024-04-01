export const lossSymbols = {
  logit_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  crossentropy_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  hinge_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  huber_loss: {
    parameters: ["f32"],
    result: "pointer",
  } as const,
  mae_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  mse_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  smooth_hinge_loss: {
    parameters: [],
    result: "pointer",
  } as const,
  tukey_loss: {
    parameters: ["f32"],
    result: "pointer",
  } as const,
};
