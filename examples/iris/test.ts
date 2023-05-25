import { parse } from "https://deno.land/std@0.188.0/csv/parse.ts";
import { LogisticRegressor } from "../../src/native/mod.ts";

const _data = Deno.readTextFileSync("examples/iris/iris.csv");

const data = parse(_data);

const ymap = ["Iris-setosa", "Iris-versicolor"]
const reg = new LogisticRegressor({ epochs: 10000, silent: true });



const x = data.map(fl => new Float32Array(fl.slice(0, 4).map(Number)))

const y = data.map(fl => ymap.indexOf(fl[4]))

reg.train(x, y)

const _data1 = Deno.readTextFileSync("examples/iris/iris_test.csv");

const data1 = parse(_data1);

const testx = data1.map(fl => new Float32Array(fl.slice(0, 4).map(Number)))
const testy = data1.map(fl => ymap.indexOf(fl[4]))

testx.forEach((fl, i) => {
    const yp = reg.predict(fl)
    console.log(fl, "expected", testy[i] === 0 ? "negative" : "positive", "got", yp)
 })