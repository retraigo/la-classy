import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";

import { CLASSY_LALA_VERSION } from "../../../version.ts";
import symbols from "./symbols/mod.ts";

const options: FetchOptions = {
  name: "classy",
  url: new URL(import.meta.url).protocol !== "file:"
    ? new URL(
      `https://github.com/retraigo/classy-lala/releases/download/${CLASSY_LALA_VERSION}/`,
      import.meta.url,
    )
    : "./target/release/",
  cache: "reloadAll",
};


const classy: Deno.DynamicLibrary<typeof symbols> = await dlopen(
  options,
  symbols,
);

const cs = classy.symbols;

export default cs;