/**
 * Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
 * The dataset contains a set of records under 5 attributes -
 * Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).
 * For demonstration with Logistic Regression, only two classes are used.
 */

/** Import csv parser from the standard library */
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
/** Import Logistic Regressor (https://deno.land/x/classylala/src/native/mod.ts) */
import { LogisticRegressor } from "../../src/native/classification.ts";
import { splitData } from "../../src/helpers/split.ts";

/** Define classes */
const ymap = ["Setosa", "Versicolor"];

/** Read the training dataset */
const _data = Deno.readTextFileSync("examples/iris/iris.csv");
const data = parse(_data);

/** Train for 10000 epochs */
const reg = new LogisticRegressor({ epochs: 10000, silent: true });

/** Get the predictors (x) and classes (y) */
const x = data.map((fl) => fl.slice(0, 4).map(Number));
const y = data.map((fl) => ymap.indexOf(fl[4]));

const [train, test] = splitData(x, y, [7, 3], true);

/** Train the model with the training data */
reg.train(train[0], train[1]);

console.log("Trained Complete");

/** Check Metrics */
const cMatrix = reg.confusionMatrix(test[0], test[1]);

console.log("Confusion Matrix: ", cMatrix)
console.log("Accuracy: ", cMatrix.true / cMatrix.size);
