import { describe, expect, it } from "vitest";
import { extractPeaks, fitHomography, fitLattice } from "./latticeFit";
import type { Point } from "./quadWarp";

/** Deterministic LCG — a fixed seed keeps a failing case reproducible. */
function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => ((state = (state * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/** Projects lattice coords through a perspective-ish map standing in for a
 * camera looking at a board from an angle. */
function project(column: number, row: number, tilt: number): Point {
  const w = 1 + tilt * row;
  return { x: (100 + column * 40) / w, y: (80 + row * 40) / w };
}

function board(size: number, tilt: number): Point[] {
  const points: Point[] = [];
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) points.push(project(column, row, tilt));
  }
  return points;
}

function cornerError(fitted: readonly Point[], expected: readonly Point[]): number {
  // Rotation of the board in frame is unresolvable from geometry alone, so a
  // fit is correct if it matches the expected outline under SOME cyclic shift.
  let best = Infinity;
  for (let shift = 0; shift < 4; shift++) {
    let worst = 0;
    for (let i = 0; i < 4; i++) {
      const a = fitted[(i + shift) % 4]!;
      const b = expected[i]!;
      worst = Math.max(worst, Math.hypot(a.x - b.x, a.y - b.y));
    }
    best = Math.min(best, worst);
  }
  return best;
}

function outline(size: number, tilt: number): Point[] {
  return [
    project(-0.5, -0.5, tilt),
    project(size - 0.5, -0.5, tilt),
    project(size - 0.5, size - 0.5, tilt),
    project(-0.5, size - 0.5, tilt),
  ];
}

describe("fitHomography", () => {
  it("recovers a known projective map from four correspondences", () => {
    const from = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }];
    const to = [{ x: 12, y: 9 }, { x: 88, y: 14 }, { x: 79, y: 71 }, { x: 20, y: 66 }];
    const h = fitHomography(from, to)!;
    expect(h).not.toBeNull();

    from.forEach((point, index) => {
      const w = h[6]! * point.x + h[7]! * point.y + h[8]!;
      const x = (h[0]! * point.x + h[1]! * point.y + h[2]!) / w;
      const y = (h[3]! * point.x + h[4]! * point.y + h[5]!) / w;
      expect(x).toBeCloseTo(to[index]!.x, 4);
      expect(y).toBeCloseTo(to[index]!.y, 4);
    });
  });

  it("least-squares fits an over-determined system", () => {
    const from = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }, { x: 2, y: 2 }];
    const to = from.map((p) => ({ x: 3 * p.x + 5, y: 3 * p.y + 7 }));
    const h = fitHomography(from, to)!;
    const w = h[6]! * 2 + h[7]! * 1 + h[8]!;
    expect((h[0]! * 2 + h[1]! * 1 + h[2]!) / w).toBeCloseTo(11, 3);
    expect((h[3]! * 2 + h[4]! * 1 + h[5]!) / w).toBeCloseTo(10, 3);
  });
});

describe("fitLattice", () => {
  it("recovers a clean 5x5 board", () => {
    const fit = fitLattice(board(5, 0.06))!;
    expect(fit).not.toBeNull();
    expect(fit.gridSize).toBe(5);
    expect(fit.inlierCount).toBe(25);
    expect(cornerError(fit.quad, outline(5, 0.06))).toBeLessThan(1);
  });

  it("recovers a clean 4x4 board rather than calling it a partial 5x5", () => {
    const fit = fitLattice(board(4, 0.05))!;
    expect(fit.gridSize).toBe(4);
    expect(fit.inlierCount).toBe(16);
    expect(cornerError(fit.quad, outline(4, 0.05))).toBeLessThan(1);
  });

  it("survives missing centres — the point of using 25 peaks instead of 4 corners", () => {
    const points = board(5, 0.06);
    // Drop a corner die and a run of interior ones: glare and hands do this.
    const kept = points.filter((_, index) => ![0, 6, 7, 12, 18, 24].includes(index));
    const fit = fitLattice(kept)!;
    expect(fit.gridSize).toBe(5);
    expect(fit.inlierCount).toBe(19);
    expect(cornerError(fit.quad, outline(5, 0.06))).toBeLessThan(1.5);
  });

  it("ignores off-lattice clutter", () => {
    const random = rng(7);
    const points = [...board(5, 0.06)];
    for (let i = 0; i < 12; i++) points.push({ x: 40 + random() * 260, y: 30 + random() * 240 });
    const fit = fitLattice(points)!;
    expect(fit.gridSize).toBe(5);
    expect(fit.inlierCount).toBeGreaterThanOrEqual(25);
    expect(cornerError(fit.quad, outline(5, 0.06))).toBeLessThan(2);
  });

  it("tolerates peak jitter", () => {
    const random = rng(11);
    const points = board(5, 0.06).map((p) => ({ x: p.x + (random() - 0.5) * 5, y: p.y + (random() - 0.5) * 5 }));
    const fit = fitLattice(points)!;
    expect(fit.gridSize).toBe(5);
    expect(fit.inlierCount).toBeGreaterThanOrEqual(22);
    expect(cornerError(fit.quad, outline(5, 0.06))).toBeLessThan(6);
  });

  it("reports low fill instead of a confident wrong answer on pure clutter", () => {
    const random = rng(3);
    const points = Array.from({ length: 25 }, () => ({ x: random() * 300, y: random() * 300 }));
    const fit = fitLattice(points);
    if (fit) expect(fit.fill).toBeLessThan(0.55);
  });

  it("returns null when there is nothing to fit", () => {
    expect(fitLattice([{ x: 1, y: 1 }, { x: 2, y: 2 }])).toBeNull();
  });
});

describe("extractPeaks", () => {
  it("finds sub-cell centres of Gaussian blobs", () => {
    const size = 32;
    const heatmap = new Float32Array(size * size);
    const centres = [{ x: 8.5, y: 6.5 }, { x: 20.25, y: 17.75 }];
    for (const centre of centres) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const value = Math.exp(-(((x + 0.5 - centre.x) ** 2 + (y + 0.5 - centre.y) ** 2) / (2 * 1.1 ** 2)));
          heatmap[y * size + x] = Math.max(heatmap[y * size + x]!, value);
        }
      }
    }

    const peaks = extractPeaks(heatmap, size, size).sort((a, b) => a.x - b.x);
    expect(peaks).toHaveLength(2);
    peaks.forEach((peak, index) => {
      expect(peak.x).toBeCloseTo(centres[index]!.x, 0);
      expect(peak.y).toBeCloseTo(centres[index]!.y, 0);
    });
  });

  it("ignores everything below threshold", () => {
    const heatmap = new Float32Array(16 * 16).fill(0.05);
    expect(extractPeaks(heatmap, 16, 16)).toHaveLength(0);
  });
});
