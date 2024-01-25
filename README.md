<img src="/assets/lala.webp" alt="La Lala" height="256px" width="auto">

<h1>La Classy</h1>

Single Layer Perceptron (SLP) library for Deno.

This library is written TypeScript and Rust and it uses FFI.

## Why Classy?

- It's fast.
- It gives you some freedom to experiment with different combinations of loss functions, activation functions, etc.
- It's easy to use.

## Features

- Optimization Algorithms:
  - Gradient Descent
  - Stochastic Average Gradients
  - Ordinary Least Squares
- Optimizers for updating weights:
  - RMSProp
  - ADAM
- Schedulers for learning rate:
  - One-cycle Scheduler
  - Decay
- Regularization
- Activation Functions:
  - Linear (regression, SVM, etc.)
  - Sigmoid (logistic regression)
  - Softmax (multinomial logistic regression)
  - Tanh (it's just there)
- Loss Functions:
  - Mean Squared Error (regression)
  - Mean Absolute Error (regression)
  - Cross-Entropy (multinomial classification)
  - Binary Cross-Entropy / Logistic Loss (binary classification)
  - Hinge Loss (binary classification, SVM)

## Quick Example

### Regression
```ts
import { OLSSolver } from "https://deno.land/x/classylala/mod.ts";

const x = [100, 23, 53, 56, 12, 98, 75];
const y = x.map((a) => [a * 6 + 13, a * 4 + 2]);

const solver = new OLSSolver();

solver.train(
  { data: Float64Array.from(x), shape: [x.length, 1] },
  { data: Float64Array.from(y.flat()), shape: [y.length, 2] },
  { silent: false, fit_intercept: true }
);

const res = solver.predict({
  data: Float64Array.from(x),
  shape: [x.length, 1],
});
for (const pred of res.rows()) {
  console.log(pred);
}
```

There are other examples in [/examples](https://github.com/retraigo/la-classy/tree/main/examples)

## Documentation

[Deno `/x`](https://deno.land/x/classylala/mod.ts)

## Maintainers

Pranev ([retraigo](https://github.com/retraigo))

Check out [deno-ml](https://github.com/retraigo/deno-ml) for examples!
Discord: [Kuro's ~~Chaos Abyss~~ Graveyard](https://discord.gg/A69vvdK)
