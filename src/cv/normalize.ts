/**
 * Per-tile photometric normalisation for classifier input.
 *
 * Motivation, measured on the 1372 reviewed crops: accuracy falls from ~83% on
 * the brightest fifth of crops to 58.5% on the darkest, while being flat
 * across contrast (76.0/74.9/77.8/74.2/71.7). So the lever is absolute
 * brightness, not contrast — a plain contrast stretch has nothing to grab.
 *
 * Every function maps a grayscale [0,1] tile to a grayscale [0,1] tile of the
 * same size, so a mode can be swapped in without touching model shape. Any
 * mode used at inference MUST also be applied when training — see the skew
 * warning in preprocess.ts.
 */

export type NormalizeMode = "none" | "standardize" | "stretch" | "local" | "bandpass";

/** Zero mean, unit variance, then recentred on 0.5. Removes exposure and gain
 * outright; keeps the glyph's polarity and relative shape. */
export function standardize(pixels: Float32Array): Float32Array {
  let mean = 0;
  for (const value of pixels) mean += value;
  mean /= pixels.length;

  let variance = 0;
  for (const value of pixels) variance += (value - mean) ** 2;
  const deviation = Math.sqrt(variance / pixels.length) || 1e-6;

  const out = new Float32Array(pixels.length);
  // 0.25 keeps a typical +/-2 sigma inside [0,1] instead of clipping the
  // glyph's own extremes, which is where the letter identity lives.
  for (let i = 0; i < pixels.length; i++) out[i] = clamp(0.5 + 0.25 * ((pixels[i]! - mean) / deviation));
  return out;
}

/** Percentile contrast stretch — robust to a few blown or black pixels. */
export function stretch(pixels: Float32Array, lowPercentile = 0.02, highPercentile = 0.98): Float32Array {
  const sorted = Float32Array.from(pixels).sort();
  const low = sorted[Math.floor(lowPercentile * (sorted.length - 1))]!;
  const high = sorted[Math.floor(highPercentile * (sorted.length - 1))]!;
  const span = high - low || 1e-6;

  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = clamp((pixels[i]! - low) / span);
  return out;
}

/** Separable box blur, used as the local-mean estimate. */
function boxBlur(pixels: Float32Array, size: number, radius: number): Float32Array {
  const horizontal = new Float32Array(pixels.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      let count = 0;
      for (let dx = -radius; dx <= radius; dx++) {
        const sx = x + dx;
        if (sx < 0 || sx >= size) continue;
        sum += pixels[y * size + sx]!;
        count++;
      }
      horizontal[y * size + x] = sum / count;
    }
  }

  const out = new Float32Array(pixels.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= size) continue;
        sum += horizontal[sy * size + x]!;
        count++;
      }
      out[y * size + x] = sum / count;
    }
  }
  return out;
}

/** Divide out the local illumination: flattens a shadow falling across one
 * side of a die, which a whole-tile normalisation cannot do. */
export function localNormalize(pixels: Float32Array, size: number, radius = 6): Float32Array {
  const local = boxBlur(pixels, size, radius);
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = clamp(0.5 + (pixels[i]! - local[i]!) * 2);
  return out;
}

/** Difference of box blurs: drops the illumination gradient (low band) and the
 * sensor noise (high band), keeping stroke-width structure. */
export function bandpass(pixels: Float32Array, size: number, inner = 1, outer = 6): Float32Array {
  const fine = boxBlur(pixels, size, inner);
  const coarse = boxBlur(pixels, size, outer);
  const out = new Float32Array(pixels.length);
  for (let i = 0; i < pixels.length; i++) out[i] = clamp(0.5 + (fine[i]! - coarse[i]!) * 2.5);
  return out;
}

function clamp(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

export function normalizeTile(pixels: Float32Array, size: number, mode: NormalizeMode): Float32Array {
  switch (mode) {
    case "standardize": return standardize(pixels);
    case "stretch": return stretch(pixels);
    case "local": return localNormalize(pixels, size);
    case "bandpass": return bandpass(pixels, size);
    default: return pixels;
  }
}
