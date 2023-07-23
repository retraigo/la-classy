/**
 * Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
 * The dataset contains text messages that are marked spam and ham (not spam).
 */

//  Import csv parser from the standard library 
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import Logistic Regressor (https://deno.land/x/classylala/src/native/mod.ts) 
import { LogisticRegressor } from "../../src/native/classification/logistic_regression.ts";
// Import CountVectorizer to convert text into vectors 
import { CountVectorizer, TfIdfTransformer } from "https://deno.land/x/vectorizer@v0.0.2/mod.ts";
import { splitData } from "../../src/helpers/split.ts";

// Define classes 
const ymap = ["spam", "ham"];

// Read the training dataset 
const _data = Deno.readTextFileSync("examples/spam/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Get the classes
const y = data.map((msg) => ymap.indexOf(msg[0]));

const [train, test] = splitData(x, y, [7, 3], true)

// Vectorize the text messages
const vec = new CountVectorizer({ stopWords: "english", lowercase: true }).fit(train[0]);
const x_vec = vec.transform(train[0])
const tfidf = new TfIdfTransformer().fit(x_vec)

const x_tfidf = tfidf.transform(x_vec)

// Initialize logistic regressor and train for 100 epochs
const reg = new LogisticRegressor({ epochs: 200, silent: false });
reg.train(x_tfidf, train[1]);

const xvec_test = tfidf.transform(vec.transform(test[0]))

// Test for accuracy
console.log("Trained Complete");

/** Check Metrics */
const cMatrix = reg.confusionMatrix(xvec_test, test[1]);

console.log("Confusion Matrix: ", cMatrix)
console.log("Accuracy: ", `${(cMatrix.true / cMatrix.size) * 10}%`);
