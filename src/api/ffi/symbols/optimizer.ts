export const optimizerSymbols = {
    adam_optimizer: {
      parameters: [
        "f32",
        "f32",
        "f32",
        "usize",
        "usize",
      ],
      result: "pointer",
    } as const,
    rmsprop_optimizer: {
      parameters: [
        "f32",
        "f32",
        "usize",
        "usize",
      ],
      result: "pointer",
    } as const,
    no_optimizer: {
      parameters: [],
      result: "pointer",
    } as const,
  };