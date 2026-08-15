import { describe, expect, it } from "vitest";
import { detectGrid, sampleRegion, type HeatmapPredictor } from "./gridDetector";
import type { PixelBuffer, Point } from "./quadWarp";

const INPUT = 128;
const OUTPUT = 32;

function blank(width: number, height: number, value = 0): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * 4);
  data.fill(value);
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  return { width, height, data };
}

/** Stands in for the trained model: renders peaks wherever the crop it is
 * shown actually contains die centres, so the scan's geometry is what is
 * under test rather than any learned behaviour. */
function fakeModel(centres: readonly Point[], source: PixelBuffer, calls: { region: number[] }[] = []): HeatmapPredictor {
  // The crop is recoverable from the sampled pixels only if we mark it, so
  // instead the caller tracks regions via the closure below.
  let index = 0;
  const regions = regionSequence(source);

  function renderInto(heatmaps: Float32Array, slot: number, region: number[]): void {
    const base = slot * OUTPUT * OUTPUT;
    for (const centre of centres) {
      const x = ((centre.x - region[0]!) / region[2]!) * OUTPUT;
      const y = ((centre.y - region[1]!) / region[3]!) * OUTPUT;
      if (x < 0 || x >= OUTPUT || y < 0 || y >= OUTPUT) continue;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const px = Math.round(x - 0.5) + dx;
          const py = Math.round(y - 0.5) + dy;
          if (px < 0 || px >= OUTPUT || py < 0 || py >= OUTPUT) continue;
          const value = Math.exp(-(((px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2) / (2 * 1.1 ** 2)));
          const at = base + py * OUTPUT + px;
          heatmaps[at] = Math.max(heatmaps[at]!, value);
        }
      }
    }
  }

  return async (_batch, count) => {
    const heatmaps = new Float32Array(count * OUTPUT * OUTPUT);
    for (let slot = 0; slot < count; slot++) {
      const region = regions[index++] ?? regions[regions.length - 1]!;
      calls.push({ region });
      renderInto(heatmaps, slot, region);
    }
    return heatmaps;
  };
}

/** Mirrors detectGrid's own crop order so the fake model knows what it is
 * being shown. Kept deliberately simple and independent of the implementation
 * it checks, apart from the shared default scales. */
function regionSequence(source: PixelBuffer): number[][] {
  const shorter = Math.min(source.width, source.height);
  // First look is a centred square, not the whole frame: resampling a
  // non-square rect into a square input squashes the dice.
  const regions: number[][] = [
    [(source.width - shorter) / 2, (source.height - shorter) / 2, shorter, shorter],
  ];
  for (const scale of [1, 0.5, 0.25]) {
    const side = shorter * scale;
    const step = Math.max(1, side * 0.5);
    const lastX = Math.max(0, source.width - side);
    const lastY = Math.max(0, source.height - side);
    const columns = Math.max(1, Math.ceil(lastX / step) + 1);
    const rows = Math.max(1, Math.ceil(lastY / step) + 1);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        regions.push([
          columns === 1 ? 0 : (column / (columns - 1)) * lastX,
          rows === 1 ? 0 : (row / (rows - 1)) * lastY,
          side,
          side,
        ]);
      }
    }
  }
  return regions;
}

function boardCentres(originX: number, originY: number, pitch: number): Point[] {
  const centres: Point[] = [];
  for (let row = 0; row < 5; row++) {
    for (let column = 0; column < 5; column++) centres.push({ x: originX + column * pitch, y: originY + row * pitch });
  }
  return centres;
}

describe("sampleRegion", () => {
  it("resamples a sub-rectangle to grayscale", () => {
    const source = blank(64, 64, 0);
    // A bright 16x16 block at (32,32): sampling exactly that block must be white.
    for (let y = 32; y < 48; y++) {
      for (let x = 32; x < 48; x++) {
        const o = (y * 64 + x) * 4;
        source.data[o] = 255; source.data[o + 1] = 255; source.data[o + 2] = 255;
      }
    }
    const patch = sampleRegion(source, { x: 32, y: 32, width: 16, height: 16 }, 8);
    expect(Math.min(...patch)).toBeGreaterThan(0.9);

    const elsewhere = sampleRegion(source, { x: 0, y: 0, width: 16, height: 16 }, 8);
    expect(Math.max(...elsewhere)).toBeLessThan(0.1);
  });

  it("clamps samples to the source bounds", () => {
    const source = blank(32, 32, 128);
    const patch = sampleRegion(source, { x: -10, y: -10, width: 60, height: 60 }, 8);
    expect(patch.every((value) => value > 0.4 && value < 0.6)).toBe(true);
  });
});

describe("detectGrid", () => {
  it("finds a frame-filling board in the first pass or two", async () => {
    const source = blank(400, 300);
    const centres = boardCentres(120, 70, 40);
    const calls: { region: number[] }[] = [];
    const fit = await detectGrid(source, fakeModel(centres, source, calls), { refine: false });

    expect(fit).not.toBeNull();
    expect(fit!.gridSize).toBe(5);
    expect(fit!.inlierCount).toBe(25);
    // Dispatches, not crops: a level is evaluated in one batched call, so the
    // scan now finishes a level it could once abandon halfway. That costs
    // extra crops and saves round-trips, which is the trade being made.
    expect(fit!.batches).toBeLessThanOrEqual(2);
  });

  it("finds a small board that a single full-frame look cannot resolve", async () => {
    // 5% of frame: below the model's measured working range, so only the scan
    // can reach it. This is the case that took real recall from 22/45 to 45/45.
    const source = blank(1600, 1200);
    const centres = boardCentres(1080, 820, 28);
    const fit = await detectGrid(source, fakeModel(centres, source), { refine: false });

    expect(fit).not.toBeNull();
    expect(fit!.inlierCount).toBeGreaterThanOrEqual(24);
    expect(fit!.passes).toBeGreaterThan(2);

    // The quad must land on the board, not merely somewhere.
    const centreX = fit!.quad.reduce((sum, p) => sum + p.x, 0) / 4;
    const centreY = fit!.quad.reduce((sum, p) => sum + p.y, 0) / 4;
    expect(Math.hypot(centreX - (1080 + 2 * 28), centreY - (820 + 2 * 28))).toBeLessThan(28);
  });

  it("never resamples a non-square rect into the square model input", async () => {
    // The bug this pins: the first look used to be the whole frame, and
    // sampleRegion squashes whatever rect it is given into a square. On
    // 9:16 phone video that stretched every die 1.78:1 and the live scanner
    // found the board in 3 frames out of 42.
    const source = blank(1080, 1920);
    const regions: { region: number[] }[] = [];
    await detectGrid(source, fakeModel([], source, regions), { refine: false });

    expect(regions.length).toBeGreaterThan(0);
    for (const { region } of regions) {
      const [, , width, height] = region;
      expect(width).toBeCloseTo(height!, 6);
    }
  });

  it("spends far fewer dispatches than crops once it has to descend", async () => {
    // The deep scan is dispatch-bound, so what matters for latency is the
    // predictor-call count, not the crop count.
    const source = blank(1600, 1200);
    const centres = boardCentres(1080, 820, 28);
    const fit = await detectGrid(source, fakeModel(centres, source), { refine: false });

    expect(fit).not.toBeNull();
    expect(fit!.batches).toBeLessThan(fit!.passes);
    expect(fit!.batches).toBeLessThanOrEqual(Math.ceil(fit!.passes / 4));
  });

  it("returns null rather than a quad when there is no board", async () => {
    const source = blank(400, 300);
    expect(await detectGrid(source, fakeModel([], source), { refine: false })).toBeNull();
  });
});
