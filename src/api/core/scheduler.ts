import symbols from "../ffi/ffi.ts";

/**
 * Apply linear decay scheduling for learning rate.
 * @param rate Rate of decay
 * @param step_size Number of steps between each decay.
 */
export function linearDecay(
  rate: number,
  step_size: number,
): Deno.PointerValue {
  return symbols.linear_decay_scheduler(rate, step_size);
}
/**
 * Apply exponential decay scheduling for learning rate.
 * @param rate Rate of decay
 * @param step_size Number of steps between each decay.
 */
export function exponentialDecay(
  rate: number,
  step_size: number,
): Deno.PointerValue {
  return symbols.exponential_decay_scheduler(rate, step_size);
}
/**
 * Apply one cycle scheduling for learning rate.
 * Make sure to set learning rate to a very small value while training.
 * @param max_rate Maximum value the learning rate can get to.
 * @param step_size Number of steps between each decay.
 */
export function oneCycle(
  max_rate: number,
  step_size: number,
): Deno.PointerValue {
  return symbols.one_cycle_scheduler(max_rate, step_size);
}

/**
 * Apply no scheduling
 */
export function noDecay(): Deno.PointerValue {
  return symbols.no_decay_scheduler();
}
