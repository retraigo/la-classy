export const optimizerSymbols = {
    adam_optimizer: {
      parameters: [
        "f64",
        "f64",
        "f64",
        "usize",
      ],
      result: "pointer",
    } as const,
    no_optimizer: {
      parameters: [],
      result: "pointer",
    } as const,
  };