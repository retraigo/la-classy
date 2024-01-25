import { useSeries } from "https://deno.land/x/vectorizer@v0.3.4/mod.ts";
import { OLSSolver } from "../src/mod.ts";

const x = Float64Array.from(useSeries(12, 120, 2));
const y = x.map((a) => a * 6 + 13);

const solver = new OLSSolver();

solver.train(
  { data: x, shape: [x.length, 1] },
  { data: y, shape: [y.length, 1] },
  true,
  false
);

console.log(solver.weights)