/**
 * Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
 * The dataset contains text messages that are marked spam and ham (not spam).
 */

//  Import csv parser from the standard library 
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
// Import Logistic Regressor (https://deno.land/x/classylala/src/native/mod.ts) 
import { LogisticRegressor } from "../../src/native/mod.ts";
// Import CountVectorizer to convert text into vectors 
import { CountVectorizer, TfIdfTransformer } from "https://deno.land/x/vectorizer@v0.0.2/mod.ts";

// Define classes 
const ymap = ["spam", "ham"];

// Read the training dataset 
const _data = Deno.readTextFileSync("examples/spam/spam.csv");
const data = parse(_data);

// Get the predictors (messages)
const x = data.map((msg) => msg[1]);

// Vectorize the text messages
const vec = new CountVectorizer({ stopWords: "english", lowercase: true }).fit(x);
const x_vec = vec.transform(x)
const tfidf = new TfIdfTransformer().fit(x_vec)

const x_tfidf = tfidf.transform(x_vec)

// Get the classes
const y = data.map((msg) => ymap.indexOf(msg[0]));

// Initialize logistic regressor and train for 100 epochs
const reg = new LogisticRegressor({ epochs: 200, silent: true });
reg.train(x_tfidf, y);

// Read the testing dataset
const _data1 = Deno.readTextFileSync("examples/spam/spam_test.csv");
const data1 = parse(_data1);

// Get the predictors and classes
const testx = vec.transform(data1.map((msg) => msg[1]));
const testy = data1.map((msg) => ymap.indexOf(msg[0]));

// Test for accuracy
let acc = 0
testx.forEach((fl, i) => {
  const yp = reg.predict(tfidf.transform([fl])[0]);
  if(yp === testy[i]) acc += 1
  // Uncomment for logs
  /*
  console.log(
    "expected",
    ymap[testy[i]],
    "got",
    ymap[yp],
  );
  */
});
console.log("Accuracy: ", (acc/testx.length))
/*
function parse(d: string): string[][] {
  return d.split("\n").map((line) => {
    const m = line.indexOf(" ");
    return [line.slice(0, m), line.slice(m + 1)];
  });
}
*/