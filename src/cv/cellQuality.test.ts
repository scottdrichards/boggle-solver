import { describe, expect, it } from "vitest";
import { assessCellQuality } from "./cellQuality";
import type { PixelBuffer } from "./quadWarp";

function solidBuffer(width: number, height: number, rgb: [number, number, number]): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set([...rgb, 255], i);
  return { width, height, data };
}

/** A crop that's mostly a mid-grey glyph with a bright specular patch covering `fraction`. */
function withGlarePatch(width: number, height: number, fraction: number): PixelBuffer {
  const buffer = solidBuffer(width, height, [120, 120, 120]);
  const glarePixels = Math.floor(width * height * fraction);
  for (let i = 0; i < glarePixels; i++) buffer.data.set([255, 255, 255, 255], i * 4);
  return buffer;
}

describe("assessCellQuality", () => {
  it("accepts a normal-contrast mid-brightness crop", () => {
    const data = new Uint8ClampedArray(32 * 32 * 4);
    for (let i = 0; i < data.length; i += 4) {
      const v = i % 8 === 0 ? 60 : 180; // checkerboard-ish, real contrast
      data.set([v, v, v, 255], i);
    }
    const quality = assessCellQuality({ width: 32, height: 32, data });
    expect(quality.rejected).toBe(false);
  });

  it("rejects a crop with a large blown-highlight patch", () => {
    const quality = assessCellQuality(withGlarePatch(32, 32, 0.3));
    expect(quality.glareFraction).toBeGreaterThan(0.12);
    expect(quality.rejected).toBe(true);
  });

  it("does not reject a small glare speck below the fraction limit", () => {
    const quality = assessCellQuality(withGlarePatch(32, 32, 0.02));
    expect(quality.rejected).toBe(false);
  });

  it("rejects a flat, bright, washed-out crop even with no single blown pixel", () => {
    const quality = assessCellQuality(solidBuffer(32, 32, [200, 200, 200]));
    expect(quality.stdDev).toBe(0);
    expect(quality.rejected).toBe(true);
  });

  it("does not reject a flat, dark crop — that's just a shadowed die, not glare", () => {
    const quality = assessCellQuality(solidBuffer(32, 32, [60, 60, 60]));
    expect(quality.rejected).toBe(false);
  });
});
