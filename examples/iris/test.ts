/**
 * Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
 * The dataset contains a set of records under 5 attributes -
 * Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).
 * For demonstration with Logistic Regression, only two classes are used.
 */

/** Import csv parser from the standard library */
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
/** Import Logistic Regressor (https://deno.land/x/classylala/src/native/mod.ts) */
import { LogisticRegressor } from "../../src/native/mod.ts";

/** Define classes */
const ymap = ["Iris-setosa", "Iris-versicolor"];

/** Read the training dataset */
const _data = Deno.readTextFileSync("examples/iris/iris.csv");
const data = parse(_data);

/** Train for 10000 epochs */
const reg = new LogisticRegressor({ epochs: 10000, silent: true });

/** Get the predictors (x) and classes (y) */
const x = data.map((fl) => new Float32Array(fl.slice(0, 4).map(Number)));
const y = data.map((fl) => ymap.indexOf(fl[4]));

/** Train the model with the training data */
reg.train(x, y);

/** Read the testing dataset */
const _data1 = Deno.readTextFileSync("examples/iris/iris_test.csv");
const data1 = parse(_data1);

/** Get the predictors and classes */
const testx = data1.map((fl) => new Float32Array(fl.slice(0, 4).map(Number)));
const testy = data1.map((fl) => ymap.indexOf(fl[4]));

/** Get the accuracy score */
let acc = 0;
testx.forEach((fl, i) => {
  const yp = reg.predict(fl);
  if (yp === testy[i]) acc += 1;
  // uncomment this line to test it live
  // console.log("expected", ymap[testy[i]], "got", ymap[yp])
});
console.log("Accuracy: ", acc / testx.length);
