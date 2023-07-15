export const logistic_regression_sym = {
    logistic_regression: {
      parameters: [
        "buffer",
        "buffer",
        "usize",
        "usize",
        "usize",
        "usize",
        "usize",
      ],
      result: "pointer",
    } as const,
    logistic_regression_predict_y: {
      parameters: ["pointer", "buffer"],
      result: "f32",
    } as const,
    logistic_regression_free_res: {
      parameters: ["pointer"],
      result: "void",
    } as const,
  };