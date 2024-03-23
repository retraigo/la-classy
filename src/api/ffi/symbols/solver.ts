export const solverSymbols = {
  gd_solver: {
    parameters: ["pointer", "pointer", "pointer", "pointer"],
    result: "pointer",
  } as const,
  sag_solver: {
    parameters: ["pointer", "pointer", "pointer", "pointer"],
    result: "pointer",
  } as const,
  ols_solver: {
    parameters: [],
    result: "pointer",
  } as const,
  solve: {
    parameters: [
      "buffer",
      "buffer",
      "buffer",
      "usize",
      "usize",
      "usize",
      "usize",
      "f32",
      "bool",
      "usize",
      "bool",
      "f32",
      "isize",
      "pointer",
      "pointer",
    ],
    result: "void",
  } as const,
  predict: {
    parameters: [
      "buffer",
      "buffer",
      "buffer",
      "usize",
      "usize",
      "usize",
      "bool",
      "pointer",
    ],
    result: "void",
  } as const,
};
