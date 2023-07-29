export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}