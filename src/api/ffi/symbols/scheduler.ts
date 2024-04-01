export const schedulerSymbols = {
  linear_decay_scheduler: {
    parameters: [
      "f32",
      "usize",
    ],
    result: "pointer",
  } as const,
  exponential_decay_scheduler: {
    parameters: [
      "f32",
      "usize",
    ],
    result: "pointer",
  } as const,
  one_cycle_scheduler: {
    parameters: [
      "f32",
      "usize",
    ],
    result: "pointer",
  } as const,
  no_decay_scheduler: {
    parameters: [],
    result: "pointer",
  } as const,
};
