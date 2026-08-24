import type { PixelBuffer } from "./quadWarp";

/**
 * Flags a cell crop as too glare-blown or washed-out to trust a
 * classification from, so it can be dropped before voting rather than
 * silently supplying a confidently-wrong letter.
 *
 * This is deliberately not a training-time fix: exposure augmentation was
 * already tried and lost on both the clean holdout and the 175 real
 * submitted cells (see CLAUDE.md, 2026-08-13) even though the brightness
 * correlation with errors is real. Rejecting bad *inputs* at inference time
 * is a different mechanism — it costs a vote, not model behaviour, and slots
 * into the existing "no confident letter" abstention path in
 * `consensus.ts`/`scanner.ts` (a frame that reads nothing is a gap in the
 * evidence, not evidence against the cell).
 */
export interface CellQuality {
  readonly meanBrightness: number;
  readonly stdDev: number;
  /** Share of pixels at or above the glare threshold. */
  readonly glareFraction: number;
  readonly rejected: boolean;
}

/** Grayscale value (0-255) above which a pixel counts as a blown highlight. */
const GLARE_PIXEL_THRESHOLD = 250;
/** A specular hotspot covering this much of the cell is enough to corrupt the glyph. */
const GLARE_FRACTION_LIMIT = 0.12;
/** Below this stddev the crop is flat — no legible glyph edge. */
const LOW_CONTRAST_STD_DEV = 18;
/** Flat *and* bright is glare wash-out; flat and dark is just a shadowed die, not rejected. */
const WASHED_OUT_BRIGHTNESS_FLOOR = 150;

export function assessCellQuality(cell: PixelBuffer): CellQuality {
  const n = cell.width * cell.height;
  let sum = 0;
  let sumSq = 0;
  let glareCount = 0;

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const gray = 0.299 * cell.data[o]! + 0.587 * cell.data[o + 1]! + 0.114 * cell.data[o + 2]!;
    sum += gray;
    sumSq += gray * gray;
    if (gray >= GLARE_PIXEL_THRESHOLD) glareCount++;
  }

  const meanBrightness = n === 0 ? 0 : sum / n;
  const variance = n === 0 ? 0 : sumSq / n - meanBrightness * meanBrightness;
  const stdDev = Math.sqrt(Math.max(0, variance));
  const glareFraction = n === 0 ? 0 : glareCount / n;

  const overexposed = glareFraction >= GLARE_FRACTION_LIMIT;
  const washedOut = stdDev < LOW_CONTRAST_STD_DEV && meanBrightness > WASHED_OUT_BRIGHTNESS_FLOOR;

  return { meanBrightness, stdDev, glareFraction, rejected: overexposed || washedOut };
}
