import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";

import { CLASSY_LALA_VERSION } from "../../../version.ts";
import { Solver } from "../types.ts";

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
  solve: {
    parameters: [
      "buffer",
      "buffer",
      "buffer",
      "usize",
      "usize",
      "buffer",
      "usize",
      "usize",
    ],
    result: "void",
  } as const,
};

const classy: Deno.DynamicLibrary<typeof symbols> = await dlopen(
  options,
  symbols,
);

const cs = classy.symbols;

export const linear = {
  solve: cs.solve as (
    weights: Uint8Array,
    data: Uint8Array,
    targets: Uint8Array,
    samples: number,
    features: number,
    config: Uint8Array,
    configLength: number,
    solver: Solver,
  ) => void,
};
