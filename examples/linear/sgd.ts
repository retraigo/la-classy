import { SgdLinearRegressor } from "../../src/native/regression.ts";
/** Import csv parser from the standard library */
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
import { splitData } from "../../src/helpers/split.ts";

/** Read the training dataset */
const _data = Deno.readTextFileSync("examples/linear/Student_Performance.csv");
const data = parse(_data);
/** Get the predictors (x) and classes (y) */
const x = data.map((fl) => [fl[0], fl[1], fl[2] === "Yes" ? 1 : 0, fl[3], fl[4]].map(Number));
const y = data.map((fl) => Number(fl[5]));
const [train, test] = splitData(x, y, [7, 3], true);

const reg = new SgdLinearRegressor({ epochs: 100000, silent: false, learningRate: 0.0005 })

reg.train(train[0], train[1])


let err = 0;
test[0].forEach((xi, i) => {
    const y_test = reg.predict(xi)
    err += (test[1][i] - y_test) ** 2
})
console.log("RMSE: ", Math.sqrt(err / test[0].length))