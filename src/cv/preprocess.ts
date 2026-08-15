import type { PixelBuffer } from "./quadWarp";

/**
 * Converts a cropped cell image into the exact format the trained
 * classifier expects: grayscale, normalized to [0,1], resized to
 * `outputSize` x `outputSize`.
 *
 * MUST stay numerically consistent with training/src/pixelOps.ts's
 * `toGrayscaleF32` + `resizeBilinear` (same luminance weights, same
 * bilinear resize) and the /255 normalization in
 * training/src/synthesize.ts — any mismatch here is a silent
 * training/serving skew bug, not a crash, so it won't be obvious from
 * errors alone if the two drift apart.
 */
export function preprocessCellForModel(cell: PixelBuffer, outputSize: number): Float32Array {
  const resized = resizeBilinear(cell, outputSize, outputSize);
  const pixels = new Float32Array(outputSize * outputSize);
  for (let i = 0; i < pixels.length; i++) {
    const o = i * 4;
    const gray = 0.299 * resized.data[o]! + 0.587 * resized.data[o + 1]! + 0.114 * resized.data[o + 2]!;
    pixels[i] = gray / 255;
  }
  return pixels;
}

function resizeBilinear(source: PixelBuffer, outWidth: number, outHeight: number): PixelBuffer {
  const data = new Uint8ClampedArray(outWidth * outHeight * 4);
  const scaleX = source.width / outWidth;
  const scaleY = source.height / outHeight;

  for (let y = 0; y < outHeight; y++) {
    const sy = Math.min(source.height - 1.001, (y + 0.5) * scaleY - 0.5);
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(source.height - 1, y0 + 1);
    const fy = sy - y0;

    for (let x = 0; x < outWidth; x++) {
      const sx = Math.min(source.width - 1.001, (x + 0.5) * scaleX - 0.5);
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(source.width - 1, x0 + 1);
      const fx = sx - x0;

      const destIndex = (y * outWidth + x) * 4;
      for (let c = 0; c < 4; c++) {
        const topLeft = source.data[(y0 * source.width + x0) * 4 + c]!;
        const topRight = source.data[(y0 * source.width + x1) * 4 + c]!;
        const bottomLeft = source.data[(y1 * source.width + x0) * 4 + c]!;
        const bottomRight = source.data[(y1 * source.width + x1) * 4 + c]!;
        const top = topLeft * (1 - fx) + topRight * fx;
        const bottom = bottomLeft * (1 - fx) + bottomRight * fx;
        data[destIndex + c] = top * (1 - fy) + bottom * fy;
      }
    }
  }

  return { width: outWidth, height: outHeight, data };
}
