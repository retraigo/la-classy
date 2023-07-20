import { useShuffle } from "../../deps.ts";

export function splitData<Tx, Ty>(
  x: Tx[],
  y: Ty[],
  ratio: [number, number] = [7, 3],
  shuffle = false,
): [[Tx[], Ty[]], [Tx[], Ty[]]] {
  if (x.length !== y.length) throw new Error("X and Y must have equal length!");
  const idx = Math.floor(x.length * (ratio[0] / (ratio[0] + ratio[1])));
  if (!shuffle) {
    return [[x.slice(0, idx), y.slice(0, idx)], [x.slice(idx), y.slice(idx)]];
  } else {
    const shuffled = useShuffle(0, x.length);
    const x1 = shuffled.slice(0, idx);
    const x2 = shuffled.slice(idx);
    return [[x1.map((i) => x[i]), x1.map((i) => y[i])], [
      x2.map((i) => x[i]),
      x2.map((i) => y[i]),
    ]];
  }
}