import { describe, expect, it } from "vitest";
import { sliceIntoCells, squareToQuadMap, warpQuadToSquare, type PixelBuffer, type Point } from "./quadWarp";

function expectPointClose(actual: Point, expected: Point, epsilon = 1e-6): void {
  expect(Math.abs(actual.x - expected.x)).toBeLessThan(epsilon);
  expect(Math.abs(actual.y - expected.y)).toBeLessThan(epsilon);
}

describe("squareToQuadMap", () => {
  it("maps unit-square corners exactly onto the quad, for an axis-aligned square (affine case)", () => {
    const corners: [Point, Point, Point, Point] = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const map = squareToQuadMap(corners);
    expectPointClose(map(0, 0), corners[0]);
    expectPointClose(map(1, 0), corners[1]);
    expectPointClose(map(1, 1), corners[2]);
    expectPointClose(map(0, 1), corners[3]);
    expectPointClose(map(0.5, 0.5), { x: 50, y: 50 });
  });

  it("maps unit-square corners exactly onto the quad, for a genuinely skewed (non-affine) quad", () => {
    // A trapezoid: narrower at the top, as a photo of a board taken at an angle would produce.
    const corners: [Point, Point, Point, Point] = [
      { x: 30, y: 0 },
      { x: 70, y: 0 },
      { x: 100, y: 100 },
      { x: 0, y: 100 },
    ];
    const map = squareToQuadMap(corners);
    expectPointClose(map(0, 0), corners[0]);
    expectPointClose(map(1, 0), corners[1]);
    expectPointClose(map(1, 1), corners[2]);
    expectPointClose(map(0, 1), corners[3]);
  });
});

function makeQuadrantImage(halfSize: number): PixelBuffer {
  // A 2*halfSize square split into 4 solid-color quadrants: TL=red, TR=green, BR=blue, BL=white.
  const size = halfSize * 2;
  const data = new Uint8ClampedArray(size * size * 4);
  const colors = {
    tl: [255, 0, 0, 255],
    tr: [0, 255, 0, 255],
    br: [0, 0, 255, 255],
    bl: [255, 255, 255, 255],
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const isRight = x >= halfSize;
      const isBottom = y >= halfSize;
      const color = isBottom ? (isRight ? colors.br : colors.bl) : isRight ? colors.tr : colors.tl;
      const i = (y * size + x) * 4;
      data.set(color, i);
    }
  }
  return { width: size, height: size, data };
}

describe("warpQuadToSquare", () => {
  it("is a no-op resample when corners cover the whole image (identity warp)", () => {
    const source = makeQuadrantImage(4); // 8x8
    const corners: [Point, Point, Point, Point] = [
      { x: 0, y: 0 },
      { x: 8, y: 0 },
      { x: 8, y: 8 },
      { x: 0, y: 8 },
    ];
    const warped = warpQuadToSquare(source, corners, 8);

    const pixelAt = (buf: PixelBuffer, x: number, y: number) => buf.data.slice((y * buf.width + x) * 4, (y * buf.width + x) * 4 + 4);
    expect(Array.from(pixelAt(warped, 1, 1))).toEqual([255, 0, 0, 255]); // top-left quadrant
    expect(Array.from(pixelAt(warped, 6, 1))).toEqual([0, 255, 0, 255]); // top-right quadrant
    expect(Array.from(pixelAt(warped, 6, 6))).toEqual([0, 0, 255, 255]); // bottom-right quadrant
    expect(Array.from(pixelAt(warped, 1, 6))).toEqual([255, 255, 255, 255]); // bottom-left quadrant
  });

  it("agrees exactly with squareToQuadMap + naive bilinear", () => {
    // warpQuadToSquare inlines its own copy of the homography and the bilinear
    // tap, because the allocating helper version cost 328 ms/call at 640x640.
    // That duplication is the risk this test exists to cover: the inlined
    // version must stay bit-for-bit identical to the readable one, on a
    // genuinely projective quad (not just the affine identity case above).
    const source = makeQuadrantImage(16); // 32x32
    const corners: [Point, Point, Point, Point] = [
      { x: 3.5, y: 2.25 },
      { x: 28.75, y: 6.5 },
      { x: 25.5, y: 29.25 },
      { x: 1.75, y: 24.5 },
    ];
    const outSize = 24;

    const map = squareToQuadMap(corners);
    const expected = new Uint8ClampedArray(outSize * outSize * 4);
    for (let py = 0; py < outSize; py++) {
      for (let px = 0; px < outSize; px++) {
        const { x, y } = map((px + 0.5) / outSize, (py + 0.5) / outSize);
        const clampedX = Math.min(Math.max(x, 0), source.width - 1);
        const clampedY = Math.min(Math.max(y, 0), source.height - 1);
        const x0 = Math.floor(clampedX);
        const y0 = Math.floor(clampedY);
        const x1 = Math.min(x0 + 1, source.width - 1);
        const y1 = Math.min(y0 + 1, source.height - 1);
        const fx = clampedX - x0;
        const fy = clampedY - y0;
        const at = (ax: number, ay: number, c: number) => source.data[(ay * source.width + ax) * 4 + c]!;
        for (let c = 0; c < 4; c++) {
          const top = at(x0, y0, c) * (1 - fx) + at(x1, y0, c) * fx;
          const bottom = at(x0, y1, c) * (1 - fx) + at(x1, y1, c) * fx;
          expected[(py * outSize + px) * 4 + c] = top * (1 - fy) + bottom * fy;
        }
      }
    }

    const warped = warpQuadToSquare(source, corners, outSize);
    expect(Array.from(warped.data)).toEqual(Array.from(expected));
  });
});

describe("sliceIntoCells", () => {
  it("splits a square into the correct number of equally-sized cells with the right pixel data", () => {
    // 4x4 image, 2x2 grid -> four 2x2 cells. Fill with a value equal to the
    // cell index (0-3) so we can confirm each slice pulls the right region.
    const width = 4;
    const data = new Uint8ClampedArray(width * width * 4);
    for (let y = 0; y < width; y++) {
      for (let x = 0; x < width; x++) {
        const cellRow = y < 2 ? 0 : 1;
        const cellCol = x < 2 ? 0 : 1;
        const cellIndex = cellRow * 2 + cellCol;
        const i = (y * width + x) * 4;
        data.set([cellIndex * 50, cellIndex * 50, cellIndex * 50, 255], i);
      }
    }
    const square: PixelBuffer = { width, height: width, data };

    const cells = sliceIntoCells(square, 2);
    expect(cells).toHaveLength(4);
    cells.forEach((cell, index) => {
      expect(cell.width).toBe(2);
      expect(cell.height).toBe(2);
      // Every pixel in this cell should carry that cell's marker value.
      for (let i = 0; i < cell.data.length; i += 4) {
        expect(cell.data[i]).toBe(index * 50);
      }
    });
  });
});
