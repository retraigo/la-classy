export const solverSymbols = {
  gd_solver: {
    parameters: ["pointer", "pointer", "pointer", "pointer"],
    result: "pointer",
  } as const,
  sgd_solver: {
    parameters: ["pointer", "pointer", "pointer", "pointer"],
    result: "pointer",
  } as const,
  minibatch_solver: {
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
      "f64",
      "bool",
      "usize",
      "bool",
      "pointer",
      "pointer",
    ],
    result: "void",
  } as const,
};
