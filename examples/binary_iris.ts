import { parse } from "https://deno.land/std@0.204.0/csv/parse.ts";

// Import helpers for metrics
import {
  ClassificationReport,
  Matrix,
  // Split the dataset
  useSplit,
} from "https://deno.land/x/vectorizer@v0.3.3/mod.ts";
import { SGDSolver } from "../src/api/core/solver/sgd.ts";
import { binCrossEntropy } from "../src/api/core/loss.ts";
import { sigmoidActivation, tanhActivation } from "../src/api/core/activation.ts";
import { ConfusionMatrix } from "../deps.ts";

// Define classes
const classes = ["Setosa", "Versicolor"];

// Read the training dataset
const _data = Deno.readTextFileSync("examples/binary_iris.csv");
const data = parse(_data);

const X = new Matrix<"f64">(Float64Array, [data.length, 4]);

// Get the predictors (x) and targets (y)
data.forEach((fl, i) => X.setRow(i, fl.slice(0, 4).map(Number)));
const y = new Matrix(Float64Array.from(data.map((fl) => classes.indexOf(fl[4]))), [data.length]);

// Split the dataset for training and testing
const [[x_train, y_train], [x_test, y_test]] = useSplit(
  { ratio: [7, 3], shuffle: true },
  X,
  y
);
const time = performance.now();

const solver = new SGDSolver({
  loss: binCrossEntropy(),
  activation: sigmoidActivation(),
});
console.log(x_train, y_train)

solver.train(x_train as Matrix<"f64">, y_train as Matrix<"f64">, { learning_rate: 0.01, epochs: 1000, silent: false });

console.log(`training time: ${performance.now() - time}ms`);

const res = []
for (const row of (x_test as Matrix<"f64">).rows()) {
  const out = solver.weights?.dot(new Matrix(row, [1])) || -1
  res.push(out)
}
const y1 = res.map((x) => (x < 0 ? 0 : 1));
const cMatrix = new ClassificationReport((y_test as Matrix<"f64">).data, y1);
console.log("Confusion Matrix: ", cMatrix);
