import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  Matrix,
  // Split the dataset
  useSplit,
  CategoricalEncoder,
} from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import {
  GradientDescentSolver,
  rmsPropOptimizer,
  softmaxActivation,
  crossEntropy,
} from "../src/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const X = new Matrix<"f64">(Float64Array, [data.length, 4]);

// Get the predictors (x) and targets (y)
data.forEach((fl, i) => X.setRow(i, fl.slice(0, 4).map(Number)));

const y_pre = data.map((fl) => fl[4]);

const encoder = new CategoricalEncoder();

const y = encoder.fit(y_pre).transform<"f64">(y_pre, "f64");

// Split the dataset for training and testing
const [[x_train, y_train], [x_test, y_test]] = useSplit(
  { ratio: [7, 3], shuffle: true },
  X,
  y
);
const time = performance.now();

const solver = new GradientDescentSolver({
  loss: crossEntropy(),
  activation: softmaxActivation(),
  optimizer: rmsPropOptimizer(4, 3),
});
solver.train(x_train, y_train, {
  learning_rate: 0.01,
  epochs: 300,
  silent: false,
  n_batches: 20,
});
console.log(`training time: ${performance.now() - time}ms`);

const res = solver.predict(x_test);

let i = 0;
for (const row of res.rows()) {
  const max = row.reduce((acc, curr, i, arr) => (arr[acc] > curr ? acc : i), 0);
  const newR = new Array(row.length).fill(0);
  newR[max] = 1;
  res.setRow(i, newR);
  i += 1;
}

const y_pred = encoder.untransform(res);
const y_act = encoder.untransform(y_test);

for (let i = 0; i < y_pred.length; i += 1) {
  console.log(y_pred[i], y_act[i]);
}

console.log(new ClassificationReport(y_act, y_pred));
