const __dirname = new URL(".", import.meta.url).pathname;

const classy = Deno.dlopen(
  `${__dirname}/../../target/release/libclassylala.so`,
  {
    linear_regression: {
      parameters: ["buffer", "buffer", "usize", "usize"],
      result: "pointer",
    },
    linear_regression_free_res: {
      parameters: ["pointer"],
      result: "void",
    },
    linear_regression_get_intercept: {
      parameters: ["pointer"],
      result: "f32",
    },
    linear_regression_get_r2: {
      parameters: ["pointer"],
      result: "f32",
    },
    linear_regression_get_slope: {
      parameters: ["pointer"],
      result: "f32",
    },
  },
);

const cs = classy.symbols;

export const linear_regression = {
  linear_regression: cs.linear_regression,
  free_res: cs.linear_regression_free_res,
  get_intercept: cs.linear_regression_get_intercept,
  get_r2: cs.linear_regression_get_r2,
  get_slope: cs.linear_regression_get_slope,
};
