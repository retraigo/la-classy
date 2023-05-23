import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";
import { CLASSY_LALA_VERSION } from "../../version.ts";

const options: FetchOptions = {
  name: "classylala",
  url: new URL(import.meta.url).protocol !== "file:"
    ? new URL(
      `https://github.com/retraigo/classy-lala/releases/download/${CLASSY_LALA_VERSION}/`,
      import.meta.url,
    )
    : "./target/release/",
  cache: "reloadAll",
};

const symbols = {
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
    result: "f32",
  } as const,
  linear_regression_get_r2: {
    parameters: ["pointer"],
    result: "f32",
  } as const,
  linear_regression_get_slope: {
    parameters: ["pointer"],
    result: "f32",
  } as const,
};

const classy: Deno.DynamicLibrary<typeof symbols> = await dlopen(
  options,
  symbols,
);

const cs = classy.symbols;

export const linear_regression = {
  linear_regression: cs.linear_regression,
  free_res: cs.linear_regression_free_res,
  get_intercept: cs.linear_regression_get_intercept,
  get_r2: cs.linear_regression_get_r2,
  get_slope: cs.linear_regression_get_slope,
};
