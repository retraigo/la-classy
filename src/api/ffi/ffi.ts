import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";

import { CLASSY_LALA_VERSION } from "../../../version.ts";

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
  gradient_descent: {
    parameters: [
      "buffer",
      "buffer",
      "buffer",
      "usize",
      "usize",
      "usize",
      "usize",
      "usize",
      "usize",
      "buffer",
      "usize",
      "f64",
      "usize",
      "usize",
      "usize",
    ],
    result: "f64",
  } as const,
  ordinary_least_squares: {
    parameters: [
      "buffer",
      "buffer",
      "buffer",
      "usize",
      "usize",
      "usize",
      "usize",
    ],
    result: "f64",
  } as const,
};

const classy: Deno.DynamicLibrary<typeof symbols> = await dlopen(
  options,
  symbols,
);

const cs = classy.symbols;

export const linear = {
  gradientDescent: cs.gradient_descent,
  ordinaryLeastSquares: cs.ordinary_least_squares,
};
