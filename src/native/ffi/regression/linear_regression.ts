export const linear_regression_sym = {
  linear_regression: {
    parameters: ["buffer", "buffer", "usize", "usize"],
    result: "pointer",
  } as const,
  linear_regression_free_res: {
    parameters: ["pointer"],
    result: "void",
  } as const,
  linear_regression_get_intercept: {
    parameters: ["pointer"],
    result: "f64",
  } as const,
  linear_regression_get_r2: {
    parameters: ["pointer"],
    result: "f64",
  } as const,
  linear_regression_get_slope: {
    parameters: ["pointer"],
    result: "f64",
  } as const,
};
