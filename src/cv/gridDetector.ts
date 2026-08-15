/**
 * Board detection: cell-centre heatmap -> lattice fit -> board quad.
 *
 * The model has a measured working range: on the 45 reviewed photos it reads
 * the board whenever the board covers >=20% of what it is shown, and fails
 * below that — at 10% frame fill the die centres are ~1.4 heatmap pixels
 * apart, closer than the peak detector's own 3x3 neighbourhood, and most
 * peaks land on background clutter instead of dice. Handheld photos are
 * routinely down at 1-15%.
 *
 * So rather than trusting one look at the whole frame, this searches a
 * coarse-to-fine pyramid of square crops and keeps the best lattice fit. That
 * is only safe because the lattice fit is a genuine hypothesis test: a crop
 * containing no board yields few inliers, so wrong crops eliminate
 * themselves and the search cannot manufacture a confident wrong answer.
 * Scanning stops as soon as a crop returns a full board, so the common
 * frame-filling photo still costs one or two forward passes.
 *
 * A final refine pass re-crops tightly around the winning fit, which puts the
 * board at the frame-filling scale the training renders use and sharpens the
 * corners.
 */
import { extractPeaks, fitLattice, type LatticeFit, type LatticeFitOptions } from "./latticeFit";
import type { PixelBuffer, Point } from "./quadWarp";

/** Runs the heatmap model on `count` stacked grayscale `inputSize`x`inputSize`
 * images, returning `count` stacked `outputSize`x`outputSize` heatmaps.
 * Injected so this module stays free of any particular tensor runtime.
 *
 * Batched rather than one-crop-at-a-time because the pyramid below is
 * dispatch-bound, not FLOP-bound: the model is tiny (128x128 in, 32x32 out)
 * and the cost that dominates is the per-call GPU dispatch plus the blocking
 * readback of the result. Evaluating a whole pyramid level in one call turns
 * up to 70 of those round-trips into one. */
export type HeatmapPredictor = (batch: Float32Array, count: number) => Promise<Float32Array>;

export interface GridDetectorOptions extends LatticeFitOptions {
  readonly inputSize?: number;
  readonly outputSize?: number;
  readonly peakThreshold?: number;
  /** Most crops to evaluate in a single dispatch. Bounds peak memory and
   * keeps some coarse-to-fine early exit inside a large level. */
  readonly maxBatch?: number;
  /** Crop padding around the located board, as a fraction of its extent. */
  readonly cropMargin?: number;
  /** Set false to skip the refine pass (for measuring what it buys). */
  readonly refine?: boolean;
  /** Square crop sides to search, as fractions of the image's shorter side.
   * Coarse first; a board at 1% of frame needs a crop near 0.25 to clear the
   * model's 20% working range. */
  readonly scales?: readonly number[];
  /** Crop stride as a fraction of crop side. 0.5 gives 50% overlap, so a
   * board can't fall permanently across a seam. */
  readonly stride?: number;
  /** Stop scanning once a fit reaches this many inliers. */
  readonly sufficientInliers?: number;
}

/** Where a scan's wall-clock actually went. Split three ways because these
 * are the three separately-fixable costs, and this project has twice now
 * optimised the wrong one by reasoning instead of measuring. */
export interface ScanTimings {
  /** CPU: resampling crops out of the source frame. */
  sampleMs: number;
  /** GPU: the batched forward passes, including readback. */
  predictMs: number;
  /** CPU: peak extraction plus lattice fitting, per crop. */
  fitMs: number;
}

export interface GridDetection extends LatticeFit {
  /** Crop that produced the fit, in source pixels — for debugging a miss. */
  readonly region: Rect;
  /** Crops evaluated. */
  readonly passes: number;
  /** Predictor calls made. Lower than `passes` whenever batching helped, and
   * it is this number, not `passes`, that tracks wall-clock latency. */
  readonly batches: number;
  readonly timings: ScanTimings;
}

export interface Rect { x: number; y: number; width: number; height: number }

/**
 * Dice on the fitted lattice below which a quad is usually wrong. Measured on
 * 45 reviewed photos: at >=22 inliers, 39 of 40 detections put every cell in
 * the right place.
 *
 * This is deliberately also the scan's early-exit threshold. It used to stop
 * only at a *perfect* 25, while both callers were happy to trust 22 — so a
 * board reading 23 or 24 (an occluded corner, one die at an angle) made the
 * scan grind through all 88 crops to confirm what it already had at crop 2.
 * On-device that cliff cost ~700 ms per scan. Perfect-or-exhaustive is a bad
 * bargain whenever "good enough" is a published number.
 */
export const CONFIDENT_INLIERS = 22;

/** A fit together with the crop it came from. */
interface Candidate { fit: LatticeFit; region: Rect }

const DEFAULTS = {
  inputSize: 128,
  outputSize: 32,
  peakThreshold: 0.3,
  cropMargin: 0.35,
  refine: true,
  scales: [1, 0.5, 0.25],
  stride: 0.5,
  sufficientInliers: CONFIDENT_INLIERS,
  maxBatch: 24,
} as const;

/** Bilinear grayscale resample of `rect` into a `size`x`size` buffer.
 *
 * The four taps are written out longhand rather than iterated. That looks
 * like premature micro-optimisation and is not: the readable version built a
 * throwaway array-of-tuples *per output pixel*, which is ~82k allocations per
 * crop, and a full pyramid does this 89 times. Measured at 2.94 ms/crop —
 * ~260 ms of a scan on a desktop CPU and the dominant cost of a deep scan on
 * a phone, against ~105 ms for all the GPU work in the same scan.
 *
 * Accumulation order is kept identical to the loop it replaced so results are
 * bit-for-bit the same. */
export function sampleRegion(source: PixelBuffer, rect: Rect, size: number): Float32Array {
  const out = new Float32Array(size * size);
  const { data, width, height } = source;

  for (let y = 0; y < size; y++) {
    const sy = Math.min(height - 1, Math.max(0, rect.y + ((y + 0.5) / size) * rect.height - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(height - 1, y0 + 1);
    const fy = sy - y0;
    const rowTop = y0 * width;
    const rowBottom = y1 * width;

    for (let x = 0; x < size; x++) {
      const sx = Math.min(width - 1, Math.max(0, rect.x + ((x + 0.5) / size) * rect.width - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(width - 1, x0 + 1);
      const fx = sx - x0;

      const o00 = (rowTop + x0) * 4;
      const o10 = (rowTop + x1) * 4;
      const o01 = (rowBottom + x0) * 4;
      const o11 = (rowBottom + x1) * 4;

      const g00 = 0.299 * data[o00]! + 0.587 * data[o00 + 1]! + 0.114 * data[o00 + 2]!;
      const g10 = 0.299 * data[o10]! + 0.587 * data[o10 + 1]! + 0.114 * data[o10 + 2]!;
      const g01 = 0.299 * data[o01]! + 0.587 * data[o01 + 1]! + 0.114 * data[o01 + 2]!;
      const g11 = 0.299 * data[o11]! + 0.587 * data[o11 + 1]! + 0.114 * data[o11 + 2]!;

      let value = (1 - fx) * (1 - fy) * g00;
      value += fx * (1 - fy) * g10;
      value += (1 - fx) * fy * g01;
      value += fx * fy * g11;
      out[y * size + x] = value / 255;
    }
  }
  return out;
}

/** Square crops covering the image, grouped into coarsest-first levels: one
 * cheap first look, then one level per scale.
 *
 * That first look is a **centred square**, not the whole frame. It used to be
 * the whole frame, and on anything far from square that was actively harmful:
 * `sampleRegion` resamples whatever rect it is given into a square 128x128, so
 * a 788x1400 phone-video frame was squashed 1.78:1 and every die arrived as a
 * tall rectangle the model has never seen. Measured on-device, the live
 * scanner — which relied on this look alone — found the board in **3 frames
 * out of 42**, then paid for repeated full pyramid scans to recover. A centred
 * square is also exactly right for a scanner: the user points at the board.
 *
 * For an already-square input this is the identical crop, so nothing changes.
 *
 * Grouped rather than flat so the scan can evaluate a whole level in one
 * batched dispatch and still stop before descending to the next — the
 * coarse-to-fine early exit is what keeps the common frame-filling photo at
 * one forward pass. */
function candidateLevels(width: number, height: number, scales: readonly number[], stride: number): Rect[][] {
  const firstLook = Math.min(width, height);
  const levels: Rect[][] = [
    [{ x: (width - firstLook) / 2, y: (height - firstLook) / 2, width: firstLook, height: firstLook }],
  ];

  const shorter = Math.min(width, height);
  for (const scale of scales) {
    const side = shorter * scale;
    if (side < 16) continue;
    const step = Math.max(1, side * stride);
    const lastX = Math.max(0, width - side);
    const lastY = Math.max(0, height - side);
    // Ceil the counts so the final row/column always reaches the far edge
    // instead of leaving an unsearched strip.
    const columns = Math.max(1, Math.ceil(lastX / step) + 1);
    const rows = Math.max(1, Math.ceil(lastY / step) + 1);

    const level: Rect[] = [];
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        level.push({
          x: columns === 1 ? 0 : (column / (columns - 1)) * lastX,
          y: rows === 1 ? 0 : (row / (rows - 1)) * lastY,
          width: side,
          height: side,
        });
      }
    }
    levels.push(level);
  }
  return levels;
}

/** Tight square crop around a fitted quad, padded by `margin`. */
function regionAround(quad: readonly Point[], margin: number, bounds: Rect): Rect {
  const xs = quad.map((point) => point.x);
  const ys = quad.map((point) => point.y);
  const centreX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centreY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const half = (Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2) * (1 + margin);

  const x = Math.max(bounds.x, centreX - half);
  const y = Math.max(bounds.y, centreY - half);
  return {
    x,
    y,
    width: Math.min(bounds.x + bounds.width, centreX + half) - x,
    height: Math.min(bounds.y + bounds.height, centreY + half) - y,
  };
}

function fitIn(heatmap: Float32Array, rect: Rect, options: Required<Pick<GridDetectorOptions, "outputSize" | "peakThreshold">> & LatticeFitOptions) {
  const peaks = extractPeaks(heatmap, options.outputSize, options.outputSize, options.peakThreshold);
  const fit = fitLattice(peaks, options);
  if (!fit) return { peaks, fit: null };

  // Peak space is outputSize cells across the crop; map back to source pixels.
  const toSource = (point: Point): Point => ({
    x: rect.x + (point.x / options.outputSize) * rect.width,
    y: rect.y + (point.y / options.outputSize) * rect.height,
  });
  return {
    peaks,
    fit: {
      ...fit,
      quad: fit.quad.map(toSource) as [Point, Point, Point, Point],
      cellCentres: fit.cellCentres.map(toSource),
    },
  };
}

export async function detectGrid(
  source: PixelBuffer,
  predict: HeatmapPredictor,
  options: GridDetectorOptions = {},
): Promise<GridDetection | null> {
  const inputSize = options.inputSize ?? DEFAULTS.inputSize;
  const outputSize = options.outputSize ?? DEFAULTS.outputSize;
  const peakThreshold = options.peakThreshold ?? DEFAULTS.peakThreshold;
  const cropMargin = options.cropMargin ?? DEFAULTS.cropMargin;
  const refine = options.refine ?? DEFAULTS.refine;
  const scales = options.scales ?? DEFAULTS.scales;
  const stride = options.stride ?? DEFAULTS.stride;
  const sufficient = options.sufficientInliers ?? DEFAULTS.sufficientInliers;
  const maxBatch = Math.max(1, options.maxBatch ?? DEFAULTS.maxBatch);
  const fitOptions = { ...options, outputSize, peakThreshold };

  const bounds: Rect = { x: 0, y: 0, width: source.width, height: source.height };
  let best: Candidate | null = null;
  let passes = 0;
  let batches = 0;
  const timings: ScanTimings = { sampleMs: 0, predictMs: 0, fitMs: 0 };

  const heatmapLength = outputSize * outputSize;

  /** Evaluates a group of crops in one dispatch, returning the best fit among
   * them. Returns rather than mutating `best` so the caller keeps ordinary
   * control flow over it — assigning a closed-over variable from in here
   * would defeat narrowing at every later use. */
  async function evaluate(regions: readonly Rect[]): Promise<Candidate | null> {
    const sampleStart = performance.now();
    const batch = new Float32Array(regions.length * inputSize * inputSize);
    regions.forEach((region, i) => {
      batch.set(sampleRegion(source, region, inputSize), i * inputSize * inputSize);
    });
    timings.sampleMs += performance.now() - sampleStart;

    const predictStart = performance.now();
    const heatmaps = await predict(batch, regions.length);
    timings.predictMs += performance.now() - predictStart;
    batches++;
    passes += regions.length;

    const fitStart = performance.now();
    let bestOfBatch: Candidate | null = null;
    regions.forEach((region, i) => {
      const heatmap = heatmaps.subarray(i * heatmapLength, (i + 1) * heatmapLength);
      const { fit } = fitIn(heatmap, region, fitOptions);
      if (fit && (!bestOfBatch || fit.inlierCount > bestOfBatch.fit.inlierCount)) bestOfBatch = { fit, region };
    });
    timings.fitMs += performance.now() - fitStart;
    return bestOfBatch;
  }

  outer: for (const level of candidateLevels(source.width, source.height, scales, stride)) {
    for (let start = 0; start < level.length; start += maxBatch) {
      const candidate = await evaluate(level.slice(start, start + maxBatch));
      if (candidate && (!best || candidate.fit.inlierCount > best.fit.inlierCount)) best = candidate;
      if (best && best.fit.inlierCount >= sufficient) break outer;
    }
  }
  if (!best) return null;

  if (refine) {
    const region = regionAround(best.fit.quad, cropMargin, bounds);
    if (region.width >= 16 && region.height >= 16) {
      const sampleStart = performance.now();
      const patch = sampleRegion(source, region, inputSize);
      timings.sampleMs += performance.now() - sampleStart;

      const predictStart = performance.now();
      const heatmap = await predict(patch, 1);
      timings.predictMs += performance.now() - predictStart;
      batches++;
      passes++;

      const fitStart = performance.now();
      const { fit } = fitIn(heatmap, region, fitOptions);
      timings.fitMs += performance.now() - fitStart;
      // Accept on ties, unlike the scan above: a full board caps at
      // gridSize^2 inliers, so the refinement that sharpens the corners of an
      // already-complete fit scores equal, never higher. Still reject a
      // strictly worse one — a tight crop can clip a board the scan had right.
      if (fit && fit.inlierCount >= best.fit.inlierCount) best = { fit, region };
    }
  }

  return { ...best.fit, region: best.region, passes, batches, timings };
}
