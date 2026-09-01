import { describe, expect, it } from "vitest";
import { detectBoardQuad, labelComponents, otsuThreshold, toGrayscale } from "./autoDetect";
import type { PixelBuffer } from "./quadWarp";

function solidBuffer(width: number, height: number, rgb: [number, number, number]): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) data.set([...rgb, 255], i);
  return { width, height, data };
}

function paintRect(buffer: PixelBuffer, x0: number, y0: number, x1: number, y1: number, rgb: [number, number, number]): void {
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * buffer.width + x) * 4;
      buffer.data.set([...rgb, 255], i);
    }
  }
}

describe("toGrayscale", () => {
  it("converts known colors to their luminance", () => {
    const buffer = solidBuffer(1, 1, [255, 255, 255]);
    expect(toGrayscale(buffer)[0]).toBeCloseTo(255, 0);

    const black = solidBuffer(1, 1, [0, 0, 0]);
    expect(toGrayscale(black)[0]).toBe(0);

    const red = solidBuffer(1, 1, [255, 0, 0]);
    expect(toGrayscale(red)[0]).toBeCloseTo(76, 0); // 0.299 * 255
  });
});

describe("otsuThreshold", () => {
  it("splits a clearly bimodal histogram between the two clusters", () => {
    const gray = new Float32Array(200);
    gray.fill(20, 0, 100);
    gray.fill(230, 100, 200);
    const threshold = otsuThreshold(gray);
    // Between-class variance is flat across the empty gap (20, 230), so any
    // value in that range is a valid Otsu answer; this implementation picks
    // the lowest (t=20). The property that actually matters is that
    // thresholding with it cleanly separates the two clusters.
    expect(threshold).toBeGreaterThanOrEqual(20);
    expect(threshold).toBeLessThan(230);
  });
});

describe("labelComponents", () => {
  it("finds two disconnected blobs with correct area/bbox/centroid", () => {
    // 5x5 mask: a 2x2 block at (0,0)-(1,1), and an isolated pixel at (4,4).
    const width = 5;
    const height = 5;
    const mask = new Uint8Array(width * height);
    const set = (x: number, y: number) => (mask[y * width + x] = 1);
    set(0, 0); set(1, 0); set(0, 1); set(1, 1);
    set(4, 4);

    const { stats } = labelComponents(mask, width, height);
    expect(stats).toHaveLength(2);

    const block = stats.find((s) => s.area === 4)!;
    expect(block).toBeDefined();
    expect(block).toMatchObject({ minX: 0, maxX: 1, minY: 0, maxY: 1, centroidX: 0.5, centroidY: 0.5 });

    const dot = stats.find((s) => s.area === 1)!;
    expect(dot).toMatchObject({ minX: 4, maxX: 4, minY: 4, maxY: 4, centroidX: 4, centroidY: 4 });
  });
});

describe("detectBoardQuad", () => {
  it("fast path: finds a bright inset rectangle against a dark background", () => {
    const buffer = solidBuffer(40, 40, [20, 20, 20]);
    paintRect(buffer, 8, 8, 32, 32, [230, 230, 230]);

    const corners = detectBoardQuad(buffer);
    expect(corners).not.toBeNull();
    const [topLeft, topRight, bottomRight, bottomLeft] = corners!;
    expect(topLeft).toEqual({ x: 8, y: 8 });
    expect(topRight).toEqual({ x: 31, y: 8 });
    expect(bottomRight).toEqual({ x: 31, y: 31 });
    expect(bottomLeft).toEqual({ x: 8, y: 31 });
  });

  it("prefers a lattice of bright cell-like blobs over a single broad bright region", () => {
    const buffer = solidBuffer(120, 120, [30, 30, 30]);
    paintRect(buffer, 10, 10, 110, 110, [70, 70, 70]);

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const x = 16 + col * 22;
        const y = 16 + row * 22;
        paintRect(buffer, x, y, x + 14, y + 14, [235, 235, 235]);
      }
    }

    const corners = detectBoardQuad(buffer);
    expect(corners).not.toBeNull();
    expect(new Set(corners!.map((point) => `${Math.round(point.x)},${Math.round(point.y)}`)).size).toBe(4);

    const xs = corners!.map((point) => point.x);
    const ys = corners!.map((point) => point.y);
    expect(Math.min(...xs)).toBeLessThan(18);
    expect(Math.max(...xs)).toBeGreaterThan(92);
    expect(Math.min(...ys)).toBeLessThan(18);
    expect(Math.max(...ys)).toBeGreaterThan(92);
  });

  it("finds a small bright face lattice even when the board is a small fraction of the frame", () => {
    const buffer = solidBuffer(420, 320, [45, 45, 45]);
    paintRect(buffer, 150, 90, 240, 180, [70, 70, 70]);

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const x = 158 + col * 16;
        const y = 98 + row * 16;
        paintRect(buffer, x, y, x + 8, y + 8, [235, 235, 235]);
      }
    }

    const corners = detectBoardQuad(buffer);
    expect(corners).not.toBeNull();

    const xs = corners!.map((point) => point.x);
    const ys = corners!.map((point) => point.y);
    expect(Math.min(...xs)).toBeLessThan(165);
    expect(Math.max(...xs)).toBeGreaterThan(225);
    expect(Math.min(...ys)).toBeLessThan(105);
    expect(Math.max(...ys)).toBeGreaterThan(165);
  });

  it("finds an oblique-looking lattice with stretched visible faces and a few missing cells", () => {
    const buffer = solidBuffer(220, 180, [35, 35, 35]);
    paintRect(buffer, 55, 35, 180, 140, [65, 65, 65]);

    const visibleCells: Array<[number, number]> = [
      [0, 0], [1, 0], [2, 0],
      [0, 1],         [2, 1],
      [0, 2], [1, 2], [2, 2],
    ];
    for (const [col, row] of visibleCells) {
      const x = 68 + col * 26;
      const y = 48 + row * 24;
      paintRect(buffer, x, y, x + 18, y + 8, [235, 235, 235]);
    }

    const corners = detectBoardQuad(buffer);
    expect(corners).not.toBeNull();

    const xs = corners!.map((point) => point.x);
    const ys = corners!.map((point) => point.y);
    expect(Math.min(...xs)).toBeLessThan(75);
    expect(Math.max(...xs)).toBeGreaterThan(130);
    expect(Math.min(...ys)).toBeLessThan(55);
    expect(Math.max(...ys)).toBeGreaterThan(100);
  });

  it("sweeps nearby bright thresholds when the face lattice is dimmer than an unrelated bright distractor", () => {
    const buffer = solidBuffer(240, 180, [35, 35, 35]);
    paintRect(buffer, 5, 5, 95, 70, [235, 235, 235]);
    paintRect(buffer, 120, 50, 205, 135, [95, 95, 95]);

    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const x = 128 + col * 18;
        const y = 58 + row * 18;
        const level = (row + col) % 2 === 0 ? 126 : 142;
        paintRect(buffer, x, y, x + 9, y + 9, [level, level, level]);
      }
    }

    const corners = detectBoardQuad(buffer);
    expect(corners).not.toBeNull();

    const xs = corners!.map((point) => point.x);
    const ys = corners!.map((point) => point.y);
    expect(Math.min(...xs)).toBeGreaterThan(118);
    expect(Math.max(...xs)).toBeGreaterThan(178);
    expect(Math.min(...ys)).toBeGreaterThan(48);
    expect(Math.max(...ys)).toBeGreaterThan(108);
  });

  it("can infer the board from a lattice of dark letter-like blobs without relying on a bright face grid", () => {
    const buffer = solidBuffer(260, 200, [150, 150, 150]);
    paintRect(buffer, 70, 45, 210, 175, [180, 180, 180]);

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        const x = 90 + col * 24;
        const y = 65 + row * 22;
        paintRect(buffer, x, y, x + 7, y + 13, [20, 20, 20]);
      }
    }

    const corners = detectBoardQuad(buffer, { minBrightAreaFraction: 0.999 });
    expect(corners).not.toBeNull();

    const xs = corners!.map((point) => point.x);
    const ys = corners!.map((point) => point.y);
    expect(Math.min(...xs)).toBeLessThan(100);
    expect(Math.max(...xs)).toBeGreaterThan(175);
    expect(Math.min(...ys)).toBeLessThan(75);
    expect(Math.max(...ys)).toBeGreaterThan(145);
  });

  it("falls back to clustering dark letter-like blobs when the bright-region gate is forced to fail", () => {
    const buffer = solidBuffer(60, 60, [230, 230, 230]);
    // A rough 2x2 arrangement of small dark "letter" blobs.
    const blobPositions: [number, number][] = [
      [10, 10],
      [40, 10],
      [10, 40],
      [40, 40],
    ];
    for (const [x, y] of blobPositions) paintRect(buffer, x, y, x + 4, y + 4, [10, 10, 10]);

    const corners = detectBoardQuad(buffer, { minBrightAreaFraction: 0.999 });
    expect(corners).not.toBeNull();
    const xs = corners!.map((p) => p.x);
    const ys = corners!.map((p) => p.y);
    // The returned quad should roughly bound the blob cluster (centroids at
    // ~11.5 and ~41.5 in each axis).
    expect(Math.min(...xs)).toBeLessThan(15);
    expect(Math.max(...xs)).toBeGreaterThan(38);
    expect(Math.min(...ys)).toBeLessThan(15);
    expect(Math.max(...ys)).toBeGreaterThan(38);
  });

  it("returns null when there isn't enough signal (uniform image)", () => {
    const buffer = solidBuffer(30, 30, [128, 128, 128]);
    expect(detectBoardQuad(buffer)).toBeNull();
  });

  it("rejects degenerate dark-blob quads with repeated corners", () => {
    const buffer = solidBuffer(60, 60, [235, 235, 235]);
    paintRect(buffer, 25, 8, 29, 12, [10, 10, 10]);
    paintRect(buffer, 28, 25, 32, 29, [10, 10, 10]);
    paintRect(buffer, 28, 42, 32, 46, [10, 10, 10]);
    paintRect(buffer, 26, 44, 30, 48, [10, 10, 10]);

    expect(detectBoardQuad(buffer, { minBrightAreaFraction: 0.999 })).toBeNull();
  });
});
