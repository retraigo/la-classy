import { LogisticRegressor } from "../../src/native/mod.ts";
import { CountVectorizer } from "https://deno.land/x/vectorizer@v0.0.1/mod.ts";
import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";

const _data = Deno.readTextFileSync("examples/spam/spam.csv");

const data = parse(_data);

const ymap = ["spam", "ham"];
const reg = new LogisticRegressor({ epochs: 50, silent: true });

const x = data.map((msg) => msg[1]);

const vec = new CountVectorizer({ stopWords: "english", lowercase: true }).fit(x);

const x_vec = vec.transform(x)
const y = data.map((msg) => ymap.indexOf(msg[0]));

reg.train(x_vec, y);

const _data1 = Deno.readTextFileSync("examples/spam/spam_test.csv");

const data1 = parse(_data1);

const testx = vec.transform(data1.map((msg) => msg[1]));
const testy = data1.map((msg) => ymap.indexOf(msg[0]));

let acc = 0
testx.forEach((fl, i) => {
  const yp = reg.predict(fl);
  if(yp === testy[i]) acc += 1
  console.log(
    "expected",
    testy[i],
    "got",
    yp,
  );
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