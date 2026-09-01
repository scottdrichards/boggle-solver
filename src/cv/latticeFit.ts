/**
 * Fits a Boggle board's cell lattice to a set of detected die-centre points.
 *
 * The detector (see training/src/trainCellHeatmap.ts) emits a heatmap whose
 * peaks are die-face centres. Those peaks are noisy and incomplete: glare
 * kills some, clutter invents others, and nothing about the raw peak set says
 * which peak is which cell. This module supplies the missing global structure.
 *
 * The constraint being exploited: under a pinhole camera, the centres of an
 * NxN grid of coplanar dice are the image of an integer lattice under a
 * homography. So a hypothesis is "peak p is lattice site (0,0), peak a is
 * (1,0), peak b is (0,1)", and its score is how many of the OTHER peaks then
 * land on integer sites. Real boards score ~25; clutter essentially never
 * scores past a handful, which is what makes the inlier count a usable
 * confidence signal rather than a number that always says yes.
 *
 * Corners come out as the lattice boundary — half a cell beyond the outermost
 * centres — so the board quad is an extrapolation of half a cell from dense
 * interior evidence, instead of a direct measurement of the tray edge, which
 * is the least reliable thing in the frame.
 */
import type { Point } from "./quadWarp";

export interface LatticeFitOptions {
  /** Grid sizes to consider, largest-first preference. */
  readonly gridSizes?: readonly number[];
  /** Max residual, in cell widths, for a peak to count as on-lattice. */
  readonly tolerance?: number;
  /** Neighbours per seed peak considered when proposing a basis. */
  readonly neighbours?: number;
  /** Reject a fit whose best window is emptier than this fraction of its sites. */
  readonly minFill?: number;
}

export interface LatticeFit {
  /** Board outline in image space: top-left, top-right, bottom-right, bottom-left. */
  readonly quad: [Point, Point, Point, Point];
  readonly gridSize: number;
  /** Peaks that landed on a site of the chosen window. */
  readonly inlierCount: number;
  /** inlierCount / gridSize^2 — the honest "how much of the board did we see". */
  readonly fill: number;
  /** Mean on-lattice residual in cell widths. */
  readonly meanResidual: number;
  /** Image-space centre of every site, row-major, including sites no peak hit. */
  readonly cellCentres: Point[];
}

const DEFAULTS = {
  gridSizes: [5, 4],
  tolerance: 0.3,
  neighbours: 6,
  minFill: 0.55,
} as const;

/* ---------------------------------------------------------------- homography */

/** Row-major 3x3, mapping (x,y,1) -> (x',y',w'). */
type Matrix3 = readonly number[];

function apply(h: Matrix3, x: number, y: number): Point {
  const w = h[6]! * x + h[7]! * y + h[8]!;
  return { x: (h[0]! * x + h[1]! * y + h[2]!) / w, y: (h[3]! * x + h[4]! * y + h[5]!) / w };
}

/** Hartley normalisation: centre on the centroid, scale to mean distance sqrt(2).
 * Without it the DLT is solving with lattice coords ~1 against pixel coords
 * ~1000 in the same matrix, and the normal equations lose most of their
 * precision to the conditioning. */
function normalise(points: readonly Point[]): { matrix: Matrix3; points: Point[] } {
  const n = points.length;
  const cx = points.reduce((sum, p) => sum + p.x, 0) / n;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / n;
  const mean = points.reduce((sum, p) => sum + Math.hypot(p.x - cx, p.y - cy), 0) / n;
  const scale = mean > 1e-12 ? Math.SQRT2 / mean : 1;
  return {
    matrix: [scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1],
    points: points.map((p) => ({ x: (p.x - cx) * scale, y: (p.y - cy) * scale })),
  };
}

function multiply(a: Matrix3, b: Matrix3): Matrix3 {
  const out = new Array<number>(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += a[r * 3 + k]! * b[k * 3 + c]!;
      out[r * 3 + c] = sum;
    }
  }
  return out;
}

function invert(m: Matrix3): Matrix3 | null {
  const [a, b, c, d, e, f, g, h, i] = m as [number, number, number, number, number, number, number, number, number];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-14) return null;
  return [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
}

/** Solves the square system in place by Gaussian elimination with partial pivoting. */
function solve(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(matrix[row]![col]!) > Math.abs(matrix[pivot]![col]!)) pivot = row;
    }
    if (Math.abs(matrix[pivot]![col]!) < 1e-12) return null;
    [matrix[col], matrix[pivot]] = [matrix[pivot]!, matrix[col]!];
    [rhs[col], rhs[pivot]] = [rhs[pivot]!, rhs[col]!];

    for (let row = col + 1; row < n; row++) {
      const factor = matrix[row]![col]! / matrix[col]![col]!;
      if (factor === 0) continue;
      for (let k = col; k < n; k++) matrix[row]![k] = matrix[row]![k]! - factor * matrix[col]![k]!;
      rhs[row] = rhs[row]! - factor * rhs[col]!;
    }
  }

  const solution = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = rhs[row]!;
    for (let k = row + 1; k < n; k++) sum -= matrix[row]![k]! * solution[k]!;
    solution[row] = sum / matrix[row]![row]!;
  }
  return solution;
}

/**
 * Least-squares homography from `from` to `to` (>= 4 correspondences), via the
 * DLT normal equations with h33 fixed to 1. Fixing h33 is safe here because a
 * lattice-to-image map never sends the lattice origin to infinity.
 */
export function fitHomography(from: readonly Point[], to: readonly Point[]): Matrix3 | null {
  if (from.length < 4 || from.length !== to.length) return null;
  const source = normalise(from);
  const target = normalise(to);

  const normal = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
  const rhs = new Array<number>(8).fill(0);

  for (let i = 0; i < from.length; i++) {
    const { x, y } = source.points[i]!;
    const { x: u, y: v } = target.points[i]!;
    const rows = [
      [x, y, 1, 0, 0, 0, -u * x, -u * y],
      [0, 0, 0, x, y, 1, -v * x, -v * y],
    ];
    const values = [u, v];
    for (let r = 0; r < 2; r++) {
      const row = rows[r]!;
      for (let a = 0; a < 8; a++) {
        for (let b = 0; b < 8; b++) normal[a]![b] = normal[a]![b]! + row[a]! * row[b]!;
      }
      for (let a = 0; a < 8; a++) rhs[a] = rhs[a]! + row[a]! * values[r]!;
    }
  }

  const solution = solve(normal, rhs);
  if (!solution) return null;

  const normalised: Matrix3 = [...solution.slice(0, 6), solution[6]!, solution[7]!, 1];
  const inverseTarget = invert(target.matrix);
  if (!inverseTarget) return null;
  return multiply(inverseTarget, multiply(normalised, source.matrix));
}

/* ------------------------------------------------------------------- peaks */

/**
 * Local maxima of a heatmap, refined to sub-cell precision by the intensity
 * centroid of the 3x3 neighbourhood. The refinement matters more than it
 * looks: at a 32x32 output a whole board spans ~22 cells, so one output cell
 * is a third of a die, and rounding every peak to integers would put a
 * visible skew into the fitted homography.
 */
/**
 * Most peaks kept, strongest first.
 *
 * `fitLattice` is roughly O(peaks * neighbours^2) and additionally sorts all
 * peaks by distance once per seed, so its cost climbs steeply with the peak
 * count — and the crops that produce the *most* peaks are the noisy ones
 * containing no board, which is the overwhelming majority of a pyramid scan.
 * Measured on-device before this cap, lattice fitting was the single largest
 * phase of a live scan at 31-47% of total wall-clock (up to 6.3 s).
 *
 * A 5x5 board has 25 centres, so 64 leaves enormous headroom for partial
 * boards and duplicate detections while bounding the pathological case.
 */
const MAX_PEAKS = 64;

export function extractPeaks(
  heatmap: Float32Array,
  width: number,
  height: number,
  threshold = 0.3,
  maxPeaks = MAX_PEAKS,
): Point[] {
  const peaks: { point: Point; value: number }[] = [];
  const at = (x: number, y: number): number => heatmap[y * width + x]!;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const value = at(x, y);
      if (value < threshold) continue;

      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if ((dx || dy) && at(x + dx, y + dy) > value) { isMax = false; break; }
        }
      }
      if (!isMax) continue;

      let weight = 0;
      let sumX = 0;
      let sumY = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const w = at(x + dx, y + dy);
          weight += w;
          sumX += w * (x + dx);
          sumY += w * (y + dy);
        }
      }
      peaks.push({
        point: weight > 0 ? { x: sumX / weight + 0.5, y: sumY / weight + 0.5 } : { x: x + 0.5, y: y + 0.5 },
        value,
      });
    }
  }

  // Only pay the sort when the cap actually bites; the common case is a
  // handful of peaks and this stays a straight pass-through.
  if (peaks.length > maxPeaks) {
    peaks.sort((a, b) => b.value - a.value);
    peaks.length = maxPeaks;
  }
  return peaks.map((entry) => entry.point);
}

/* ----------------------------------------------------------------- lattice */

interface Assignment {
  readonly column: number;
  readonly row: number;
  readonly peak: Point;
  readonly residual: number;
}

/** Assigns each peak to its nearest integer site under `imageToLattice`,
 * keeping only near-integer hits and, where two peaks claim one site, the
 * closer one. */
function assign(peaks: readonly Point[], imageToLattice: Matrix3, tolerance: number): Assignment[] {
  const best = new Map<number, Assignment>();
  for (const peak of peaks) {
    const { x, y } = apply(imageToLattice, peak.x, peak.y);
    const column = Math.round(x);
    const row = Math.round(y);
    const residual = Math.hypot(x - column, y - row);
    if (residual > tolerance) continue;

    // Numeric key, not `${column},${row}`: this runs for every peak on every
    // refit round of every hypothesis — tens of thousands of times per fit —
    // and building a string each time was pure allocation. Sites are small
    // integers and may be negative, so offset into a non-negative range
    // before packing rather than relying on the sign surviving.
    const key = (row + SITE_OFFSET) * SITE_STRIDE + (column + SITE_OFFSET);
    const existing = best.get(key);
    if (!existing || residual < existing.residual) best.set(key, { column, row, peak, residual });
  }
  return [...best.values()];
}

/** Lattice sites live well within +/-512 of the origin for any real board;
 * anything beyond that is a degenerate hypothesis that scores nothing anyway. */
const SITE_OFFSET = 512;
const SITE_STRIDE = 1024;

function latticePoints(assignments: readonly Assignment[]): { lattice: Point[]; image: Point[] } {
  return {
    lattice: assignments.map((a) => ({ x: a.column, y: a.row })),
    image: assignments.map((a) => a.peak),
  };
}

/** Median distance from a peak to its nearest neighbour — the die pitch, as
 * long as most peaks are real dice. Hypothesised basis vectors are held near
 * this, which is what stops the search from "explaining" the board with a
 * double-spacing or diagonal lattice that covers more stray points and scores
 * better on raw inlier count. */
function medianSpacing(peaks: readonly Point[]): number {
  const distances = peaks.map((peak) => {
    let best = Infinity;
    for (const other of peaks) {
      if (other === peak) continue;
      best = Math.min(best, Math.hypot(other.x - peak.x, other.y - peak.y));
    }
    return best;
  });
  distances.sort((a, b) => a - b);
  return distances[Math.floor(distances.length / 2)] ?? 0;
}

/** Best-scoring `size`x`size` axis-aligned window over assigned sites. */
function bestWindow(assignments: readonly Assignment[], size: number): { column: number; row: number; inliers: Assignment[] } | null {
  if (!assignments.length) return null;
  const columns = assignments.map((a) => a.column);
  const rows = assignments.map((a) => a.row);
  let best: { column: number; row: number; inliers: Assignment[] } | null = null;

  for (let row = Math.min(...rows) - size + 1; row <= Math.max(...rows); row++) {
    for (let column = Math.min(...columns) - size + 1; column <= Math.max(...columns); column++) {
      const inliers = assignments.filter(
        (a) => a.column >= column && a.column < column + size && a.row >= row && a.row < row + size,
      );
      if (!best || inliers.length > best.inliers.length) best = { column, row, inliers };
    }
  }
  return best;
}

/** Clockwise from whichever corner sits closest to the image origin, so the
 * quad is always traversed consistently. Which physical corner that is depends
 * on how the board was rotated in frame — an ambiguity no geometric fit can
 * resolve, and one the downstream letter pass already searches over. */
function orderCorners(corners: readonly Point[]): [Point, Point, Point, Point] {
  const cx = corners.reduce((sum, p) => sum + p.x, 0) / corners.length;
  const cy = corners.reduce((sum, p) => sum + p.y, 0) / corners.length;
  const clockwise = [...corners].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );

  let start = 0;
  for (let i = 1; i < clockwise.length; i++) {
    if (Math.hypot(clockwise[i]!.x, clockwise[i]!.y) < Math.hypot(clockwise[start]!.x, clockwise[start]!.y)) start = i;
  }
  return [0, 1, 2, 3].map((i) => clockwise[(start + i) % 4]!) as [Point, Point, Point, Point];
}

/**
 * Finds the best NxN cell lattice explaining `peaks`, in whatever coordinate
 * space the peaks are given in (heatmap cells or image pixels — the fit is
 * scale-free, so the caller can rescale the returned quad).
 */
export function fitLattice(peaks: readonly Point[], options: LatticeFitOptions = {}): LatticeFit | null {
  const gridSizes = options.gridSizes ?? DEFAULTS.gridSizes;
  const tolerance = options.tolerance ?? DEFAULTS.tolerance;
  const neighbourCount = options.neighbours ?? DEFAULTS.neighbours;
  const minFill = options.minFill ?? DEFAULTS.minFill;
  if (peaks.length < 6) return null;

  const spacing = medianSpacing(peaks);
  if (!(spacing > 0)) return null;

  /** Window count for the best grid size, ties broken toward the smaller
   * board: every 4x4 sub-window of a full 5x5 also holds 16 inliers, so a
   * larger size only earns the reading if it explains strictly MORE peaks
   * *without* explaining a meaningfully smaller share of its own board.
   *
   * The second half of that rule was added 2026-08-24 after real 4x4 photos
   * (background clutter, not a synthetic render) measured the raw-count-only
   * version choosing 5x5 every time: a real 4x4 board plus a few unrelated
   * peaks from clutter (table edge, box, wood grain) explains MORE peaks
   * under a 5x5 hypothesis just because 25 sites offer more chances for
   * clutter to land within tolerance of *some* site — not because there's
   * really a 5th ring of dice. Measured on 3 real photos of one physical 4x4
   * board: the true 4x4 window always hit fill 1.00 (16/16, tight residual),
   * while the spurious 5x5 reading the old rule picked scored only 0.64-0.80
   * fill. A genuine 5x5 board's whole-board fill, by contrast, was never
   * more than ~0.04 below its own embedded-4x4-subwindow's fill (0.96 vs
   * 1.00 on a reviewed demo photo) — real additional rows are dense and
   * consistent, unlike scattered clutter. `LARGER_SIZE_FILL_SLACK` sits well
   * inside that gap: generous enough for a real larger board's few missed
   * dice, tight enough that a clutter-inflated larger reading can't clear
   * it. */
  const LARGER_SIZE_FILL_SLACK = 0.12;
  const score = (assignments: readonly Assignment[]) => {
    let best: { size: number; window: NonNullable<ReturnType<typeof bestWindow>> } | null = null;
    for (const size of [...gridSizes].sort((a, b) => a - b)) {
      const window = bestWindow(assignments, size);
      if (!window) continue;
      if (!best) {
        best = { size, window };
        continue;
      }
      const bestFill = best.window.inliers.length / (best.size * best.size);
      const fill = window.inliers.length / (size * size);
      if (window.inliers.length > best.window.inliers.length && fill >= bestFill - LARGER_SIZE_FILL_SLACK) {
        best = { size, window };
      }
    }
    return best;
  };

  let champion: { scored: NonNullable<ReturnType<typeof score>>; homography: Matrix3 } | null = null;

  /** Nothing can beat every peak landing on a full board, so once a
   * hypothesis explains that many there is no reason to keep searching.
   * Without this the loop ground through all seeds x neighbour-pairs — up to
   * ~2300 hypotheses, each with three refit rounds — *after* already holding a
   * perfect fit. On a clean 25-peak board that was 52.8 ms of pure waste, and
   * a board filling the frame is the live scanner's common case. */
  const bestPossible = Math.min(peaks.length, Math.max(...gridSizes.map((size) => size * size)));

  search: for (let seed = 0; seed < peaks.length; seed++) {
    const origin = peaks[seed]!;
    const neighbours = peaks
      .map((peak, index) => ({ peak, index, distance: Math.hypot(peak.x - origin.x, peak.y - origin.y) }))
      .filter((entry) => entry.index !== seed)
      .sort((a, b) => a.distance - b.distance)
      .slice(0, neighbourCount);

    for (const first of neighbours) {
      if (first.distance < 0.65 * spacing || first.distance > 1.5 * spacing) continue;
      for (const second of neighbours) {
        if (first.index === second.index) continue;
        if (second.distance < 0.65 * spacing || second.distance > 1.5 * spacing) continue;
        // Reject near-parallel pairs: they span no area, so the implied basis
        // is degenerate and the "lattice" it proposes is a line.
        const cross =
          (first.peak.x - origin.x) * (second.peak.y - origin.y) -
          (first.peak.y - origin.y) * (second.peak.x - origin.x);
        if (Math.abs(cross) < 0.2 * first.distance * second.distance) continue;

        // Seed an affine basis, then let the homography refit pull in the
        // perspective the basis cannot express. Two rounds is enough: the
        // first recovers most of the board, the second cleans up the far edge
        // where foreshortening is worst.
        let homography = fitHomography(
          [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }],
          [
            origin,
            first.peak,
            second.peak,
            { x: first.peak.x + second.peak.x - origin.x, y: first.peak.y + second.peak.y - origin.y },
          ],
        );
        if (!homography) continue;

        let inverse = invert(homography);
        let assignments: Assignment[] = [];
        for (let round = 0; round < 3 && inverse; round++) {
          assignments = assign(peaks, inverse, tolerance);
          if (assignments.length < 4) break;
          const { lattice, image } = latticePoints(assignments);
          const refit = fitHomography(lattice, image);
          if (!refit) break;
          homography = refit;
          inverse = invert(refit);
        }

        if (!inverse || assignments.length < 4) continue;
        // Score on the best window, not on the raw assignment count: peaks
        // that land on the lattice but outside any NxN board are clutter that
        // happens to be collinear, and must not win the hypothesis.
        const scored = score(assignments);
        if (!scored) continue;
        if (!champion || scored.window.inliers.length > champion.scored.window.inliers.length) {
          champion = { scored, homography };
          if (champion.scored.window.inliers.length >= bestPossible) break search;
        }
      }
    }
  }

  if (!champion) return null;

  const { size, window } = champion.scored;
  if (window.inliers.length / (size * size) < minFill) return null;
  // Refit on the window's inliers alone, with the window's own origin, so a
  // stray on-lattice peak outside the board cannot drag the corners.
  const local = window.inliers.map((a) => ({ ...a, column: a.column - window.column, row: a.row - window.row }));
  const homography = fitHomography(
    local.map((a) => ({ x: a.column, y: a.row })),
    local.map((a) => a.peak),
  ) ?? champion.homography;

  const corners = orderCorners([
    apply(homography, -0.5, -0.5),
    apply(homography, size - 0.5, -0.5),
    apply(homography, size - 0.5, size - 0.5),
    apply(homography, -0.5, size - 0.5),
  ]);

  const cellCentres: Point[] = [];
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) cellCentres.push(apply(homography, column, row));
  }

  return {
    quad: corners,
    gridSize: size,
    inlierCount: window.inliers.length,
    fill: window.inliers.length / (size * size),
    meanResidual: local.reduce((sum, a) => sum + a.residual, 0) / Math.max(1, local.length),
    cellCentres,
  };
}
