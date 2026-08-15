export interface Point {
  x: number;
  y: number;
}

/** width/height + RGBA data, structurally compatible with DOM `ImageData`
 * but plain enough to construct in tests without a DOM. */
export interface PixelBuffer {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

/** The eight free coefficients of a unit-square-to-quadrilateral homography:
 * `x = (a11*u + a21*v + a31) / w`, `y = (a12*u + a22*v + a32) / w`,
 * `w = a13*u + a23*v + 1`. Exposed on its own so callers outside the warp
 * (the live-scanner lock animation, which corner-pins a CSS layer onto the
 * same quad) can reuse the exact math without re-deriving it. */
export interface QuadHomography {
  a11: number;
  a21: number;
  a31: number;
  a12: number;
  a22: number;
  a32: number;
  a13: number;
  a23: number;
}

/**
 * Closed-form mapping from the unit square [0,1]x[0,1] to an arbitrary
 * quadrilateral (corners in source-image pixel space, order: top-left,
 * top-right, bottom-right, bottom-left). This is exactly the homography a
 * general 4-point DLT would produce for this case — deriving it directly
 * avoids an 8x8 linear solve. See Heckbert, "Fundamentals of Texture
 * Mapping and Image Warping" (1989), the square-to-quadrilateral case.
 */
export function quadHomography(corners: readonly [Point, Point, Point, Point]): QuadHomography {
  const [p0, p1, p2, p3] = corners;

  const dx1 = p1.x - p2.x;
  const dx2 = p3.x - p2.x;
  const dx3 = p0.x - p1.x + p2.x - p3.x;
  const dy1 = p1.y - p2.y;
  const dy2 = p3.y - p2.y;
  const dy3 = p0.y - p1.y + p2.y - p3.y;

  let a13 = 0;
  let a23 = 0;
  const denom = dx1 * dy2 - dx2 * dy1;
  const isAffine = Math.abs(dx3) < 1e-9 && Math.abs(dy3) < 1e-9;
  if (!isAffine && Math.abs(denom) > 1e-9) {
    a13 = (dx3 * dy2 - dx2 * dy3) / denom;
    a23 = (dx1 * dy3 - dx3 * dy1) / denom;
  }

  const a11 = p1.x - p0.x + a13 * p1.x;
  const a21 = p3.x - p0.x + a23 * p3.x;
  const a31 = p0.x;
  const a12 = p1.y - p0.y + a13 * p1.y;
  const a22 = p3.y - p0.y + a23 * p3.y;
  const a32 = p0.y;

  return { a11, a21, a31, a12, a22, a32, a13, a23 };
}

/**
 * Used as an inverse map for warping: for a destination pixel at normalized
 * (u,v), this gives the source pixel to sample — no matrix inversion needed.
 */
export function squareToQuadMap(corners: readonly [Point, Point, Point, Point]): (u: number, v: number) => Point {
  const { a11, a21, a31, a12, a22, a32, a13, a23 } = quadHomography(corners);

  return (u, v) => {
    const w = a13 * u + a23 * v + 1;
    return {
      x: (a11 * u + a21 * v + a31) / w,
      y: (a12 * u + a22 * v + a32) / w,
    };
  };
}

/**
 * Inverse-maps a `outSize` x `outSize` destination square from the source quad.
 *
 * The homography and the bilinear tap are inlined here rather than expressed
 * as the obvious `map(u,v)` + `sampleBilinear(...)` helpers. That is not
 * gratuitous: at the 640x640 this is called with, those helpers allocated a
 * `{x,y}` point *and* a 4-element tuple per destination pixel — about 820,000
 * allocations for one frame — and the warp ran at **328 ms/call** on a desktop
 * CPU, dominating the live scanner's whole per-frame budget. Now ~10 ms.
 *
 * The arithmetic, its ordering, and the clamping are unchanged, so output is
 * bit-for-bit identical to the readable version; `quadWarp.test.ts` pins that.
 */
export function warpQuadToSquare(
  source: PixelBuffer,
  corners: readonly [Point, Point, Point, Point],
  outSize: number,
): PixelBuffer {
  const data = new Uint8ClampedArray(outSize * outSize * 4);
  const source_ = source.data;
  const width = source.width;
  const maxX = source.width - 1;
  const maxY = source.height - 1;

  // Same closed-form square-to-quad coefficients as `squareToQuadMap`, pulled
  // out so the inner loop is pure arithmetic on locals.
  const [p0, p1, p2, p3] = corners;
  const dx1 = p1.x - p2.x;
  const dx2 = p3.x - p2.x;
  const dx3 = p0.x - p1.x + p2.x - p3.x;
  const dy1 = p1.y - p2.y;
  const dy2 = p3.y - p2.y;
  const dy3 = p0.y - p1.y + p2.y - p3.y;

  let a13 = 0;
  let a23 = 0;
  const denom = dx1 * dy2 - dx2 * dy1;
  const isAffine = Math.abs(dx3) < 1e-9 && Math.abs(dy3) < 1e-9;
  if (!isAffine && Math.abs(denom) > 1e-9) {
    a13 = (dx3 * dy2 - dx2 * dy3) / denom;
    a23 = (dx1 * dy3 - dx3 * dy1) / denom;
  }

  const a11 = p1.x - p0.x + a13 * p1.x;
  const a21 = p3.x - p0.x + a23 * p3.x;
  const a31 = p0.x;
  const a12 = p1.y - p0.y + a13 * p1.y;
  const a22 = p3.y - p0.y + a23 * p3.y;
  const a32 = p0.y;

  for (let py = 0; py < outSize; py++) {
    const v = (py + 0.5) / outSize;
    for (let px = 0; px < outSize; px++) {
      const u = (px + 0.5) / outSize;

      const w = a13 * u + a23 * v + 1;
      const x = (a11 * u + a21 * v + a31) / w;
      const y = (a12 * u + a22 * v + a32) / w;

      const clampedX = Math.min(Math.max(x, 0), maxX);
      const clampedY = Math.min(Math.max(y, 0), maxY);
      const x0 = Math.floor(clampedX);
      const y0 = Math.floor(clampedY);
      const x1 = Math.min(x0 + 1, maxX);
      const y1 = Math.min(y0 + 1, maxY);
      const fx = clampedX - x0;
      const fy = clampedY - y0;

      const o00 = (y0 * width + x0) * 4;
      const o10 = (y0 * width + x1) * 4;
      const o01 = (y1 * width + x0) * 4;
      const o11 = (y1 * width + x1) * 4;
      const i = (py * outSize + px) * 4;

      // Measured: unrolling these four iterations and hoisting the weights is
      // worth ~3%, i.e. nothing. Removing the per-pixel allocations was the
      // entire win — don't bother re-trying the unroll.
      for (let channel = 0; channel < 4; channel++) {
        const top = source_[o00 + channel]! * (1 - fx) + source_[o10 + channel]! * fx;
        const bottom = source_[o01 + channel]! * (1 - fx) + source_[o11 + channel]! * fx;
        data[i + channel] = top * (1 - fy) + bottom * fy;
      }
    }
  }

  return { width: outSize, height: outSize, data };
}

/** Crops a square `PixelBuffer` into `gridSize` x `gridSize` equal cells, row-major. */
export function sliceIntoCells(square: PixelBuffer, gridSize: number): PixelBuffer[] {
  const cellSize = Math.floor(square.width / gridSize);
  const cells: PixelBuffer[] = [];

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const data = new Uint8ClampedArray(cellSize * cellSize * 4);
      for (let y = 0; y < cellSize; y++) {
        const srcRowStart = ((row * cellSize + y) * square.width + col * cellSize) * 4;
        const destRowStart = y * cellSize * 4;
        data.set(square.data.subarray(srcRowStart, srcRowStart + cellSize * 4), destRowStart);
      }
      cells.push({ width: cellSize, height: cellSize, data });
    }
  }

  return cells;
}
