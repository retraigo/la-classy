import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";

import { CLASSY_LALA_VERSION } from "../../../version.ts";
import { LossFunction } from "../types.ts";
import { Model } from "../types.ts";
import { Optimizer } from "../types.ts";
import { LearningRateScheduler } from "../types.ts";

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
      "f32",
      "usize",
      "buffer",
      "usize",
      "buffer",
      "usize",
      "f64",
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
  gradientDescent: cs.gradient_descent as (
    w_ptr: Uint8Array,
    x_ptr: Uint8Array,
    y_ptr: Uint8Array,
    n_samples: number,
    n_features: number,
    loss: LossFunction,
    model: Model,
    c: number,
    optimizer: Optimizer,
    optimizer_options: Uint8Array,
    scheduler: LearningRateScheduler,
    scheduler_options: Uint8Array,
    fit_intercept: number,
    learning_rate: number,
    epochs: number,
    silent: number,
  ) => number,
  ordinaryLeastSquares: cs.ordinary_least_squares,
};
