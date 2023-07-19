import { useNormalArray } from "../../deps.ts";
import { LinearRegressor } from "../../src/native/regression.ts";

const x = useNormalArray(1e7, 107, 5);
const y = x.map(n => 8 * n + 5)

const reg = new LinearRegressor()

reg.train(x, y)

const x_test = useNormalArray(1e4, 120, 1);

let err = 0;
x_test.forEach(n => {
    const y_test = reg.predict(n)
    err += (8 * n + 5 - y_test) ** 2
})
console.log("RMSE: ", Math.sqrt(err / x_test.length))
console.log("Slope: ", reg.slope, "Intercept: ", reg.intercept, "R2: ", reg.r2)