import { describe, expect, it } from "vitest";
import { preprocessCellForModel } from "./preprocess";
import type { PixelBuffer } from "./quadWarp";

function solidBuffer(width: number, height: number, rgb: [number, number, number]): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set([...rgb, 255], i);
  return { width, height, data };
}

describe("preprocessCellForModel", () => {
  it("produces the requested output size", () => {
    const buffer = solidBuffer(160, 160, [128, 128, 128]);
    const pixels = preprocessCellForModel(buffer, 32);
    expect(pixels.length).toBe(32 * 32);
  });

  it("converts a solid color to its normalized luminance", () => {
    const white = preprocessCellForModel(solidBuffer(64, 64, [255, 255, 255]), 32);
    expect(white[0]).toBeCloseTo(1, 5);

    const black = preprocessCellForModel(solidBuffer(64, 64, [0, 0, 0]), 32);
    expect(black[0]).toBeCloseTo(0, 5);

    const red = preprocessCellForModel(solidBuffer(64, 64, [255, 0, 0]), 32);
    expect(red[0]).toBeCloseTo(0.299, 3); // matches training/src/pixelOps.ts's weights
  });

  it("stays within [0,1] for arbitrary input", () => {
    const data = new Uint8ClampedArray(50 * 50 * 4);
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(Math.random() * 256);
    const pixels = preprocessCellForModel({ width: 50, height: 50, data }, 32);
    for (const value of pixels) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });
});
