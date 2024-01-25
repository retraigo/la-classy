import { useSeries } from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import { OLSSolver } from "../src/mod.ts";

const x = useSeries(12, 120, 2);
const y = x.map((a) => [a * 6 + 13, a * 4 + 2]);

const solver = new OLSSolver();

solver.train(
  { data: Float64Array.from(x), shape: [x.length, 1] },
  { data: Float64Array.from(y.flat()), shape: [y.length, 2] },
  { silent: false, fit_intercept: true }
);

console.log(solver.weights);

for (const pred of solver
  .predict({ data: Float64Array.from(x), shape: [x.length, 1] })
  .rows()) {
  console.log(pred);
}
