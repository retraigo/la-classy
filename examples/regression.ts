Deno.env.set("RUST_BACKTRACE", "FULL")
import { useSeries } from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import {
  OLSSolver,
  GradientDescentSolver,
  linearActivation,
  huber,
  tukey,
  mse,
  mae,
  rmsPropOptimizer,
  adamOptimizer,
} from "../src/mod.ts";

const x = new Array(54).map(_ => Math.random());
const y = x.map((a) => [a * 6 + 13, a * 4 + 2]);
console.log(x.length, y.length)

const solver = new GradientDescentSolver({
  loss: tukey(),
  activation: linearActivation(),
  n_batches: 6,
  optimizer: adamOptimizer(2, 2)
});

solver.train(
  { data: Float64Array.from(x), shape: [x.length, 1] },
  { data: Float64Array.from(y.flat()), shape: [y.length, 2] },
  { silent: false, fit_intercept: true, learning_rate: 0.01, epochs: 200 }
);

console.log(solver.weights);

for (const pred of solver
  .predict({ data: Float64Array.from(x), shape: [x.length, 1] })
  .rows()) {
  console.log(pred);
}
