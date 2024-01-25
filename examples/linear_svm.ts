import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  Matrix,
  // Split the dataset
  useSplit,
} from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import { SagSolver, hinge, linearActivation } from "../src/mod.ts";

// Read the training dataset
const _data = Deno.readTextFileSync("examples/binary_iris.csv");
const data = parse(_data);

const X = new Matrix<"f64">(Float64Array, [data.length, 4]);

// Get the predictors (x) and targets (y)
data.forEach((fl, i) => X.setRow(i, fl.slice(0, 4).map(Number)));
const y = new Matrix<"f64">(
  Float64Array.from(data.map((fl) => (fl[4] === "Setosa" ? 1 : -1))),
  [data.length]
);

// Split the dataset for training and testing
const [[x_train, y_train], [x_test, y_test]] = useSplit(
  { ratio: [7, 3], shuffle: true },
  X,
  y
);
const time = performance.now();

const solver = new SagSolver({
  loss: hinge(),
  activation: linearActivation(),
});
solver.train(x_train, y_train, {
  learning_rate: 0.001,
  epochs: 1000,
  silent: false,
  n_batches: 0,
  patience: 5,
  tolerance: 0,
});
console.log(`training time: ${performance.now() - time}ms`);
const res = solver.predict(x_test);

let i = 0;
const y_pred = [],
  y_act = [];
for (const row of res.rows()) {
  console.log(row, y_test.row(i));
  y_act.push(y_test.row(i)[0]);
  y_pred.push(row[0] > 0 ? 1 : -1);
  i += 1;
}
console.log(new ClassificationReport(y_act, y_pred));
