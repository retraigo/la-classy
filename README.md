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
import { Matrix } from "jsr:@lala/appraisal@0.7.5";
import {
  GradientDescentSolver,
  adamOptimizer,
  huber,
} from "jsr:@lala/classy@1.2.1";

const x = [100, 23, 53, 56, 12, 98, 75];
const y = x.map((a) => [a * 6 + 13, a * 4 + 2]);

const solver = new GradientDescentSolver({
  // Huber loss is a mix of MSE and MAE
  loss: huber(),
  // ADAM optimizer with 1 + 1 input for intercept, 2 outputs.
  optimizer: adamOptimizer(2, 2),
});

// Train for 700 epochs in 2 minibatches
solver.train(
  new Matrix(
    x.map((n) => [n]),
    "f64"
  ),
  new Matrix(y, "f64"),
  { silent: false, fit_intercept: true, epochs: 700, n_batches: 2 }
);

const res = solver.predict(
  new Matrix(
    x.map((n) => [n]),
    "f64"
  )
);

for (let i = 0; i < res.nRows; i += 1) {
  console.log(Array.from(res.row(i)), y[i]);
}
```

There are other examples in [/examples](https://github.com/retraigo/la-classy/tree/main/examples)

## Documentation

[Deno `/x`](https://deno.land/x/classylala/mod.ts)

## Maintainers

Pranev ([retraigo](https://github.com/retraigo))

Check out [deno-ml](https://github.com/retraigo/deno-ml) for examples!
Discord: [Kuro's ~~Chaos Abyss~~ Graveyard](https://discord.gg/A69vvdK)
