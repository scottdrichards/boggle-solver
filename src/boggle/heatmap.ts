import type { ScoredWord } from "./score";

/**
 * Per-cell "how many points route through here": for each cell, the sum of
 * `points` over every found word whose path visits it, normalized so the
 * board's hottest cell is 1.
 *
 * A single pass over each word's path accumulates into a flat `Float64Array`
 * indexed by cell (no `Map<cell, points>`, no per-cell object) — O(total
 * letters across all found words), which is strictly less work than the
 * solver already did to find those words. A second pass finds the max and a
 * third divides by it; all three are linear scans over small typed arrays,
 * so this is cheap enough to recompute on every result set without caching.
 */
export function computeCellHeatmap(results: readonly ScoredWord[], cellCount: number): Float64Array {
  const weights = new Float64Array(cellCount);
  for (const { path, points } of results) {
    for (const cell of path) weights[cell]! += points;
  }
  let max = 0;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i]! > max) max = weights[i]!;
  }
  if (max > 0) {
    for (let i = 0; i < weights.length; i++) weights[i]! /= max;
  }
  return weights;
}
