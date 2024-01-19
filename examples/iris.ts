import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  Matrix,
  // Split the dataset
  useSplit,
  CategoricalEncoder
} from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import { binCrossEntropy } from "../src/api/core/loss.ts";
import { sigmoidActivation, tanhActivation } from "../src/api/core/activation.ts";
import {sigmoid} from "../src/helpers.ts"
import { GradientDescentSolver } from "../src/mod.ts";
import { softmaxActivation } from "../src/mod.ts";
import { crossEntropy } from "../src/mod.ts";
import { adamOptimizer } from "../src/mod.ts";
import { regularizer } from "../src/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/iris.csv");
const data = parse(_data);

// Get the predictors (x) and targets (y)
const X = new Matrix<"f64">(Float64Array, [data.length, 4]);

// Get the predictors (x) and targets (y)
data.forEach((fl, i) => X.setRow(i, fl.slice(0, 4).map(Number)));

const y_pre = data.map((fl) => fl[4]);

const encoder = new CategoricalEncoder()

const y = encoder.fit(y_pre).transform<"f64">(y_pre, "f64")

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
  optimizer: adamOptimizer(4, 3, 0.9, 0.999, 1e-14)
});
solver.train(x_train, y_train, { learning_rate: 0.01, epochs: 100, silent: false, n_batches: 20 });
console.log(`training time: ${performance.now() - time}ms`);

const res = solver.predict(x_test)

let i = 0
const y_pred = [], y_act = []
for (const row of res.rows()) {
  y_act.push(y_test.row(i).reduce((acc, curr, i, arr) => arr[acc] > curr ? acc : i, 0))
  y_pred.push(row.reduce((acc, curr, i) => row[acc] > curr ? acc : i, 0))
  i += 1
}
console.log(new ClassificationReport(y_act, y_pred))