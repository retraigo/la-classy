import { lossSymbols } from "./loss.ts";
import { optimizerSymbols } from "./optimizer.ts";
import { regularizerSymbols } from "./regularizer.ts";
import { schedulerSymbols } from "./scheduler.ts";
import { solverSymbols } from "./solver.ts";
import { activationSymbols } from "./activation.ts";

export default {
  ...lossSymbols,
  ...optimizerSymbols,
  ...regularizerSymbols,
  ...schedulerSymbols,
  ...solverSymbols,
  ...activationSymbols,
};
