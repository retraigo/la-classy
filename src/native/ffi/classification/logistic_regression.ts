export const logistic_regression_sym = {
    logistic_regression: {
      parameters: [
        "buffer",
        "buffer",
        "usize",
        "usize",
        "usize",
        "f64",
        "usize",
        "usize",
      ],
      result: "pointer",
    } as const,
    logistic_regression_confusion_matrix: {
      parameters: [
        "pointer",
        "buffer",
        "buffer",
        "usize",
        "usize",
        "buffer",
      ],
      result: "void",
    } as const,
    logistic_regression_predict_y: {
      parameters: ["pointer", "buffer"],
      result: "f64",
    } as const,
    logistic_regression_free_res: {
      parameters: ["pointer"],
      result: "void",
    } as const,
  };