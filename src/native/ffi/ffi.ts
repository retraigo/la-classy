import { dlopen, FetchOptions } from "https://deno.land/x/plug@1.0.2/mod.ts";
import {
  linear_regression_sym,
  sgd_linear_regression_sym,
} from "./regression.ts";
import { logistic_regression_sym } from "./classification.ts";
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
  ...linear_regression_sym,
  ...logistic_regression_sym,
  ...sgd_linear_regression_sym,
};

const classy: Deno.DynamicLibrary<typeof symbols> = await dlopen(
  options,
  symbols,
);

const cs = classy.symbols;

export const linear_regression = {
  train: cs.linear_regression,
  free_res: cs.linear_regression_free_res,
  get_intercept: cs.linear_regression_get_intercept,
  get_r2: cs.linear_regression_get_r2,
  get_slope: cs.linear_regression_get_slope,
};

export const sgd_linear_regression = {
  train: cs.sgd_linear_regression,
  free_res: cs.sgd_linear_regression_free_res,
  predict: cs.sgd_linear_regression_predict_y,
};

export const logistic_regression = {
  train: cs.logistic_regression,
  predict: cs.logistic_regression_predict_y,
  confusion_matrix: cs.logistic_regression_confusion_matrix,
  free_res: cs.logistic_regression_free_res,
};
