import type { PixelBuffer, Point } from "./quadWarp";

export function toGrayscale(source: PixelBuffer): Float32Array {
  const gray = new Float32Array(source.width * source.height);
  for (let i = 0; i < gray.length; i++) {
    const o = i * 4;
    gray[i] = 0.299 * source.data[o]! + 0.587 * source.data[o + 1]! + 0.114 * source.data[o + 2]!;
  }
  return gray;
}

/** Otsu's method: the threshold that maximizes between-class variance of a bimodal histogram. */
export function otsuThreshold(gray: Float32Array): number {
  const histogram = new Array<number>(256).fill(0);
  for (const value of gray) histogram[Math.min(255, Math.max(0, Math.round(value)))]!++;

  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i]!;

  let weightBackground = 0;
  let sumBackground = 0;
  let maxVariance = 0;
  let threshold = 0;

  for (let t = 0; t < 256; t++) {
    weightBackground += histogram[t]!;
    if (weightBackground === 0) continue;
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;

    sumBackground += t * histogram[t]!;
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const variance = weightBackground * weightForeground * (meanBackground - meanForeground) ** 2;

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = t;
    }
  }

  return threshold;
}

export interface ComponentStats {
  label: number;
  area: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  centroidX: number;
  centroidY: number;
}

/** 4-connected connected-component labeling via iterative flood fill (no recursion, so large regions are safe). */
export function labelComponents(
  mask: Uint8Array,
  width: number,
  height: number,
): { labels: Int32Array; stats: ComponentStats[] } {
  const labels = new Int32Array(width * height).fill(-1);
  const stats: ComponentStats[] = [];
  const stack: number[] = [];

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || labels[start]! !== -1) continue;

    const label = stats.length;
    let area = 0;
    let sumX = 0;
    let sumY = 0;
    let minX = width;
    let maxX = -1;
    let minY = height;
    let maxY = -1;

    labels[start] = label;
    stack.push(start);

    while (stack.length > 0) {
      const index = stack.pop()!;
      const x = index % width;
      const y = Math.floor(index / width);

      area++;
      sumX += x;
      sumY += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      if (x > 0) tryVisit(index - 1);
      if (x < width - 1) tryVisit(index + 1);
      if (y > 0) tryVisit(index - width);
      if (y < height - 1) tryVisit(index + width);
    }

    stats.push({ label, area, minX, maxX, minY, maxY, centroidX: sumX / area, centroidY: sumY / area });

    function tryVisit(neighbor: number): void {
      if (mask[neighbor] !== 0 && labels[neighbor] === -1) {
        labels[neighbor] = label;
        stack.push(neighbor);
      }
    }
  }

  return { labels, stats };
}

/** Finds the 4 extreme points of a point set along the +x+y/+x-y axes — exact
 * corners for an (arbitrarily rotated) rectangle, a robust cheap stand-in
 * for a full convex-hull-then-simplify pipeline at this scale. */
function extremeCorners(points: readonly Point[]): [Point, Point, Point, Point] | null {
  let topLeft: Point | undefined;
  let topRight: Point | undefined;
  let bottomRight: Point | undefined;
  let bottomLeft: Point | undefined;
  let minSum = Infinity;
  let maxSum = -Infinity;
  let minDiff = Infinity;
  let maxDiff = -Infinity;

  for (const point of points) {
    const sum = point.x + point.y;
    const diff = point.x - point.y;
    if (sum < minSum) {
      minSum = sum;
      topLeft = point;
    }
    if (sum > maxSum) {
      maxSum = sum;
      bottomRight = point;
    }
    if (diff > maxDiff) {
      maxDiff = diff;
      topRight = point;
    }
    if (diff < minDiff) {
      minDiff = diff;
      bottomLeft = point;
    }
  }

  if (!topLeft || !topRight || !bottomRight || !bottomLeft) return null;
  return [topLeft, topRight, bottomRight, bottomLeft];
}

function polygonArea(points: readonly Point[]): number {
  let twiceArea = 0;
  for (let i = 0; i < points.length; i++) {
    const current = points[i]!;
    const next = points[(i + 1) % points.length]!;
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}

function hasFourDistinctCorners(quad: readonly Point[]): boolean {
  return new Set(quad.map((point) => `${Math.round(point.x)},${Math.round(point.y)}`)).size === 4;
}

function expandedQuadFromCenter(
  quad: readonly [Point, Point, Point, Point],
  expansion: number,
  width: number,
  height: number,
): [Point, Point, Point, Point] {
  const center = quad.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 });
  center.x /= quad.length;
  center.y /= quad.length;

  return quad.map((point) => {
    const dx = point.x - center.x;
    const dy = point.y - center.y;
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) return point;
    return {
      x: Math.min(width - 1, Math.max(0, point.x + (dx / length) * expansion)),
      y: Math.min(height - 1, Math.max(0, point.y + (dy / length) * expansion)),
    };
  }) as [Point, Point, Point, Point];
}

function isPlausibleQuad(
  quad: readonly [Point, Point, Point, Point] | null,
  width: number,
  height: number,
  minAreaFraction = 0.05,
  maxAreaFraction = 0.9,
): quad is [Point, Point, Point, Point] {
  if (!quad || !hasFourDistinctCorners(quad)) return false;
  const areaFraction = polygonArea(quad) / (width * height);
  return areaFraction >= minAreaFraction && areaFraction <= maxAreaFraction;
}

function extremeCornersOfLabel(labels: Int32Array, label: number, width: number, height: number): [Point, Point, Point, Point] | null {
  const points: Point[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (labels[y * width + x] === label) points.push({ x, y });
    }
  }
  return extremeCorners(points);
}

export interface DetectQuadOptions {
  /** Minimum fraction of the image area the largest bright region must cover to be trusted as the board. */
  minBrightAreaFraction?: number;
}

const DEFAULT_MIN_BRIGHT_AREA_FRACTION = 0.15;
// A region covering nearly the whole frame isn't a distinguishable board
// against a background — it's most likely a blank/uniform photo where Otsu
// had no real contrast to split on.
const MAX_BRIGHT_AREA_FRACTION = 0.97;
const MIN_BOARD_ASPECT = 0.5;
const MAX_BOARD_ASPECT = 2;
const MIN_LETTER_BLOB_AREA_FRACTION = 0.0005;
const MAX_LETTER_BLOB_AREA_FRACTION = 0.02;
const MIN_LETTER_BLOBS_FOR_FALLBACK = 4;
const MIN_DARK_LETTER_AREA_FRACTION = 0.00004;
const MAX_DARK_LETTER_AREA_FRACTION = 0.01;
const MIN_DARK_LETTER_FILL = 0.08;
const MAX_DARK_LETTER_ASPECT = 4;
const MIN_DARK_CLUSTER_SIZE = 8;
const DARK_NEIGHBOR_DISTANCE_FACTOR = 3.6;
const DARK_QUAD_EXPANSION_FACTOR = 0.9;
const MIN_FACE_AREA_FRACTION = 0.0001;
const MAX_FACE_AREA_FRACTION = 0.02;
const MIN_FACE_FILL = 0.24;
const MIN_FACE_CLUSTER_SIZE = 6;
const MAX_FACE_ASPECT = 2.6;
const MAX_CLUSTER_ASPECT = 3;
const FACE_NEIGHBOR_DISTANCE_FACTOR = 2.4;
const FACE_QUAD_EXPANSION_FACTOR = 0.5;
const MIN_SMALL_BOARD_AREA_FRACTION = 0.02;
const DARK_GRID_THRESHOLD_OFFSETS = [-36, -18, 0, 18] as const;
const FACE_GRID_THRESHOLD_OFFSETS = [-36, -18, 0, 18, 36] as const;

interface FaceGridDetection {
  quad: [Point, Point, Point, Point];
  score: number;
}

interface LetterGridDetection {
  quad: [Point, Point, Point, Point];
  score: number;
}

interface BrightFaceCandidate {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  centroidX: number;
  centroidY: number;
  width: number;
  height: number;
  area: number;
}

interface DarkLetterCandidate {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  centroidX: number;
  centroidY: number;
  width: number;
  height: number;
  area: number;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1]! + sorted[middle]!) / 2 : sorted[middle]!;
}

function average(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function relativeSpread(values: readonly number[]): number {
  const middle = median(values);
  if (middle <= 1e-6) return Infinity;
  return average(values.map((value) => Math.abs(value - middle))) / middle;
}

function nearestNeighborDistances(points: readonly Point[]): number[] {
  if (points.length < 2) return [];
  return points.map((point, index) => {
    let best = Infinity;
    for (let otherIndex = 0; otherIndex < points.length; otherIndex++) {
      if (index === otherIndex) continue;
      const other = points[otherIndex]!;
      const distance = Math.hypot(point.x - other.x, point.y - other.y);
      if (distance < best) best = distance;
    }
    return best;
  });
}

function faceGridScore(cluster: readonly BrightFaceCandidate[], quad: readonly [Point, Point, Point, Point], width: number, height: number): number {
  const bboxWidth = Math.max(...cluster.map((candidate) => candidate.maxX)) - Math.min(...cluster.map((candidate) => candidate.minX)) + 1;
  const bboxHeight = Math.max(...cluster.map((candidate) => candidate.maxY)) - Math.min(...cluster.map((candidate) => candidate.minY)) + 1;
  const bboxAspect = bboxWidth / bboxHeight;
  const bboxArea = bboxWidth * bboxHeight;
  const quadAreaFraction = polygonArea(quad) / (width * height);
  const fillScore = average(cluster.map((candidate) => candidate.area / (candidate.width * candidate.height)));
  const sizeSpreadPenalty = relativeSpread(cluster.map((candidate) => Math.sqrt(candidate.area)));
  const quadFill = bboxArea > 0 ? polygonArea(quad) / bboxArea : 0;
  const aspectPenalty = Math.abs(Math.log(Math.max(bboxAspect, 1 / bboxAspect)));
  return cluster.length * 3 + fillScore * 4 + quadFill * 2 + quadAreaFraction * 8 - sizeSpreadPenalty * 4 - aspectPenalty * 2;
}

function letterGridScore(cluster: readonly DarkLetterCandidate[], quad: readonly [Point, Point, Point, Point], width: number, height: number): number {
  const points = cluster.map((candidate) => ({ x: candidate.centroidX, y: candidate.centroidY }));
  const nearest = nearestNeighborDistances(points);
  const bboxWidth = Math.max(...cluster.map((candidate) => candidate.maxX)) - Math.min(...cluster.map((candidate) => candidate.minX)) + 1;
  const bboxHeight = Math.max(...cluster.map((candidate) => candidate.maxY)) - Math.min(...cluster.map((candidate) => candidate.minY)) + 1;
  const bboxAspect = bboxWidth / bboxHeight;
  const bboxArea = bboxWidth * bboxHeight;
  const quadAreaFraction = polygonArea(quad) / (width * height);
  const fillScore = average(cluster.map((candidate) => candidate.area / (candidate.width * candidate.height)));
  const sizeSpreadPenalty = relativeSpread(cluster.map((candidate) => Math.sqrt(candidate.area)));
  const spacingPenalty = relativeSpread(nearest);
  const quadFill = bboxArea > 0 ? polygonArea(quad) / bboxArea : 0;
  const aspectPenalty = Math.abs(Math.log(Math.max(bboxAspect, 1 / bboxAspect)));
  return cluster.length * 2.5 + fillScore * 3 + quadFill * 1.5 + quadAreaFraction * 8 - sizeSpreadPenalty * 3 - spacingPenalty * 4 - aspectPenalty * 1.5;
}

function detectLetterGridQuad(stats: readonly ComponentStats[], width: number, height: number, totalArea: number): LetterGridDetection | null {
  const letterCandidates: DarkLetterCandidate[] = stats
    .map((component) => {
      const boxWidth = component.maxX - component.minX + 1;
      const boxHeight = component.maxY - component.minY + 1;
      const aspect = boxWidth / boxHeight;
      const fill = component.area / (boxWidth * boxHeight);
      const areaFraction = component.area / totalArea;
      if (
        areaFraction < MIN_DARK_LETTER_AREA_FRACTION ||
        areaFraction > MAX_DARK_LETTER_AREA_FRACTION ||
        aspect < 1 / MAX_DARK_LETTER_ASPECT ||
        aspect > MAX_DARK_LETTER_ASPECT ||
        fill < MIN_DARK_LETTER_FILL
      ) {
        return null;
      }

      return {
        minX: component.minX,
        maxX: component.maxX,
        minY: component.minY,
        maxY: component.maxY,
        centroidX: component.centroidX,
        centroidY: component.centroidY,
        width: boxWidth,
        height: boxHeight,
        area: component.area,
      };
    })
    .filter((candidate): candidate is DarkLetterCandidate => candidate !== null);

  if (letterCandidates.length < MIN_DARK_CLUSTER_SIZE) return null;

  const medianLetterSize = median(letterCandidates.map((candidate) => Math.sqrt(candidate.area)));
  if (medianLetterSize <= 0) return null;

  const adjacency = letterCandidates.map(() => [] as number[]);
  const maxNeighborDistance = medianLetterSize * DARK_NEIGHBOR_DISTANCE_FACTOR;
  for (let i = 0; i < letterCandidates.length; i++) {
    for (let j = i + 1; j < letterCandidates.length; j++) {
      const a = letterCandidates[i]!;
      const b = letterCandidates[j]!;
      const distance = Math.hypot(a.centroidX - b.centroidX, a.centroidY - b.centroidY);
      if (distance <= maxNeighborDistance) {
        adjacency[i]!.push(j);
        adjacency[j]!.push(i);
      }
    }
  }

  const visited = new Array(letterCandidates.length).fill(false);
  let best: LetterGridDetection | null = null;

  for (let start = 0; start < letterCandidates.length; start++) {
    if (visited[start]) continue;

    const stack = [start];
    visited[start] = true;
    const cluster: DarkLetterCandidate[] = [];

    while (stack.length > 0) {
      const index = stack.pop()!;
      cluster.push(letterCandidates[index]!);
      for (const neighbor of adjacency[index]!) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          stack.push(neighbor);
        }
      }
    }

    if (cluster.length < MIN_DARK_CLUSTER_SIZE) continue;

    const minX = Math.min(...cluster.map((candidate) => candidate.minX));
    const maxX = Math.max(...cluster.map((candidate) => candidate.maxX));
    const minY = Math.min(...cluster.map((candidate) => candidate.minY));
    const maxY = Math.max(...cluster.map((candidate) => candidate.maxY));
    const aspect = (maxX - minX + 1) / (maxY - minY + 1);
    if (aspect < 1 / MAX_CLUSTER_ASPECT || aspect > MAX_CLUSTER_ASPECT) continue;

    const points = cluster.map((candidate) => ({ x: candidate.centroidX, y: candidate.centroidY }));
    const rawQuad = extremeCorners(points);
    if (!rawQuad) continue;

    const neighborDistances = nearestNeighborDistances(points);
    const medianNeighborDistance = median(neighborDistances);
    if (medianNeighborDistance <= 0) continue;

    const expandedQuad = expandedQuadFromCenter(rawQuad, medianNeighborDistance * DARK_QUAD_EXPANSION_FACTOR, width, height);
    if (!isPlausibleQuad(expandedQuad, width, height, MIN_SMALL_BOARD_AREA_FRACTION, 0.9)) continue;

    const score = letterGridScore(cluster, expandedQuad, width, height);
    if (!best || score > best.score) best = { quad: expandedQuad, score };
  }

  return best;
}

function detectFaceGridQuad(stats: readonly ComponentStats[], width: number, height: number, totalArea: number): FaceGridDetection | null {
  const faceCandidates: BrightFaceCandidate[] = stats
    .map((component) => {
      const boxWidth = component.maxX - component.minX + 1;
      const boxHeight = component.maxY - component.minY + 1;
      const aspect = boxWidth / boxHeight;
      const fill = component.area / (boxWidth * boxHeight);
      const areaFraction = component.area / totalArea;
      if (
        areaFraction < MIN_FACE_AREA_FRACTION ||
        areaFraction > MAX_FACE_AREA_FRACTION ||
        aspect < 1 / MAX_FACE_ASPECT ||
        aspect > MAX_FACE_ASPECT ||
        fill < MIN_FACE_FILL
      ) {
        return null;
      }

      return {
        minX: component.minX,
        maxX: component.maxX,
        minY: component.minY,
        maxY: component.maxY,
        centroidX: component.centroidX,
        centroidY: component.centroidY,
        width: boxWidth,
        height: boxHeight,
        area: component.area,
      };
    })
    .filter((candidate): candidate is BrightFaceCandidate => candidate !== null);

  if (faceCandidates.length < MIN_FACE_CLUSTER_SIZE) return null;

  const medianFaceSize = median(faceCandidates.map((candidate) => Math.sqrt(candidate.area)));
  if (medianFaceSize <= 0) return null;

  const adjacency = faceCandidates.map(() => [] as number[]);
  const maxNeighborDistance = medianFaceSize * FACE_NEIGHBOR_DISTANCE_FACTOR;
  for (let i = 0; i < faceCandidates.length; i++) {
    for (let j = i + 1; j < faceCandidates.length; j++) {
      const a = faceCandidates[i]!;
      const b = faceCandidates[j]!;
      const distance = Math.hypot(a.centroidX - b.centroidX, a.centroidY - b.centroidY);
      if (distance <= maxNeighborDistance) {
        adjacency[i]!.push(j);
        adjacency[j]!.push(i);
      }
    }
  }

  const visited = new Array(faceCandidates.length).fill(false);
  let best: FaceGridDetection | null = null;

  for (let start = 0; start < faceCandidates.length; start++) {
    if (visited[start]) continue;

    const stack = [start];
    visited[start] = true;
    const cluster: BrightFaceCandidate[] = [];

    while (stack.length > 0) {
      const index = stack.pop()!;
      cluster.push(faceCandidates[index]!);
      for (const neighbor of adjacency[index]!) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          stack.push(neighbor);
        }
      }
    }

    if (cluster.length < MIN_FACE_CLUSTER_SIZE) continue;

    const minX = Math.min(...cluster.map((candidate) => candidate.minX));
    const maxX = Math.max(...cluster.map((candidate) => candidate.maxX));
    const minY = Math.min(...cluster.map((candidate) => candidate.minY));
    const maxY = Math.max(...cluster.map((candidate) => candidate.maxY));
    const aspect = (maxX - minX + 1) / (maxY - minY + 1);
    if (aspect < 1 / MAX_CLUSTER_ASPECT || aspect > MAX_CLUSTER_ASPECT) continue;

    const points: Point[] = [];
    for (const candidate of cluster) {
      points.push(
        { x: candidate.minX, y: candidate.minY },
        { x: candidate.maxX, y: candidate.minY },
        { x: candidate.maxX, y: candidate.maxY },
        { x: candidate.minX, y: candidate.maxY },
      );
    }

    const rawQuad = extremeCorners(points);
    if (!rawQuad) continue;

    const expandedQuad = expandedQuadFromCenter(rawQuad, medianFaceSize * FACE_QUAD_EXPANSION_FACTOR, width, height);
    if (!isPlausibleQuad(expandedQuad, width, height, MIN_SMALL_BOARD_AREA_FRACTION, 0.9)) continue;

    const score = faceGridScore(cluster, expandedQuad, width, height);
    if (!best || score > best.score) {
      best = { quad: expandedQuad, score };
    }
  }

  return best;
}

function buildBinaryMask(gray: Float32Array, threshold: number, mode: "bright" | "dark"): Uint8Array {
  const mask = new Uint8Array(gray.length);
  if (mode === "bright") {
    for (let i = 0; i < gray.length; i++) mask[i] = gray[i]! > threshold ? 1 : 0;
  } else {
    for (let i = 0; i < gray.length; i++) mask[i] = gray[i]! <= threshold ? 1 : 0;
  }
  return mask;
}

function clampThreshold(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

/**
 * Coarse board-quad auto-detection, in two passes:
 *  1. Fast path: the largest bright connected region (a light tray/tile
 *     background against a typically-darker table), if it's big and
 *     roughly square enough to plausibly be the board.
 *  2. Fallback: cluster of small dark blobs (letters) against a lighter
 *     background — their extreme corners approximate the board's.
 * Returns null when neither pass finds enough signal; callers should fall
 * back to a manual default (e.g. an inset rectangle for the user to drag).
 */
function detectBoardQuadAtWorkingResolution(source: PixelBuffer, options: DetectQuadOptions = {}): [Point, Point, Point, Point] | null {
  const minBrightAreaFraction = options.minBrightAreaFraction ?? DEFAULT_MIN_BRIGHT_AREA_FRACTION;
  const { width, height } = source;
  const totalArea = width * height;
  const gray = toGrayscale(source);
  const threshold = otsuThreshold(gray);

  let bestLetterGrid: LetterGridDetection | null = null;
  const attemptedDarkThresholds = new Set<number>();
  for (const offset of DARK_GRID_THRESHOLD_OFFSETS) {
    const candidateThreshold = clampThreshold(threshold + offset);
    if (attemptedDarkThresholds.has(candidateThreshold)) continue;
    attemptedDarkThresholds.add(candidateThreshold);

    const darkMask = buildBinaryMask(gray, candidateThreshold, "dark");
    const darkAtThreshold = labelComponents(darkMask, width, height);
    const letterGrid = detectLetterGridQuad(darkAtThreshold.stats, width, height, totalArea);
    if (letterGrid && (!bestLetterGrid || letterGrid.score > bestLetterGrid.score)) bestLetterGrid = letterGrid;
  }

  let bestFaceGrid: FaceGridDetection | null = null;
  const attemptedThresholds = new Set<number>();
  for (const offset of FACE_GRID_THRESHOLD_OFFSETS) {
    const candidateThreshold = clampThreshold(threshold + offset);
    if (attemptedThresholds.has(candidateThreshold)) continue;
    attemptedThresholds.add(candidateThreshold);

    const brightMask = buildBinaryMask(gray, candidateThreshold, "bright");
    const brightAtThreshold = labelComponents(brightMask, width, height);
    const faceGrid = detectFaceGridQuad(brightAtThreshold.stats, width, height, totalArea);
    if (faceGrid && (!bestFaceGrid || faceGrid.score > bestFaceGrid.score)) bestFaceGrid = faceGrid;
  }

  // A repeated set of die faces is more structurally specific than
  // thresholded letter fragments, which are easily mimicked by table grain
  // and other background details. Keep letters as the color-independent
  // fallback for boards whose faces do not separate from their tray.
  if (bestFaceGrid) return bestFaceGrid.quad;
  if (bestLetterGrid) return bestLetterGrid.quad;

  const brightMask = buildBinaryMask(gray, threshold, "bright");
  const bright = labelComponents(brightMask, width, height);

  let largestBright: ComponentStats | null = null;
  for (const stats of bright.stats) {
    if (!largestBright || stats.area > largestBright.area) largestBright = stats;
  }

  const brightFraction = largestBright ? largestBright.area / totalArea : 0;
  if (largestBright && brightFraction >= minBrightAreaFraction && brightFraction <= MAX_BRIGHT_AREA_FRACTION) {
    const boxWidth = largestBright.maxX - largestBright.minX + 1;
    const boxHeight = largestBright.maxY - largestBright.minY + 1;
    const aspect = boxWidth / boxHeight;
    if (aspect >= MIN_BOARD_ASPECT && aspect <= MAX_BOARD_ASPECT) {
      const corners = extremeCornersOfLabel(bright.labels, largestBright.label, width, height);
      if (isPlausibleQuad(corners, width, height)) return corners;
    }
  }

  // Inclusive on this side so every pixel lands in exactly one of the two
  // masks — otherwise a value that lands exactly on the threshold (common
  // with an imbalanced histogram, since Otsu can return a threshold equal
  // to one cluster's own value) would be excluded from both.
  const darkMask = buildBinaryMask(gray, threshold, "dark");
  const dark = labelComponents(darkMask, width, height);

  const minBlobArea = totalArea * MIN_LETTER_BLOB_AREA_FRACTION;
  const maxBlobArea = totalArea * MAX_LETTER_BLOB_AREA_FRACTION;
  const letterBlobs = dark.stats.filter((s) => s.area >= minBlobArea && s.area <= maxBlobArea);
  if (letterBlobs.length < MIN_LETTER_BLOBS_FOR_FALLBACK) return null;

  const corners = extremeCorners(letterBlobs.map((s) => ({ x: s.centroidX, y: s.centroidY })));
  return isPlausibleQuad(corners, width, height, 0.02, 0.9) ? corners : null;
}

const MAX_DETECTION_DIMENSION = 512;

function downsampleForDetection(source: PixelBuffer, scale: number): PixelBuffer {
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const data = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    const sourceY = Math.min(source.height - 1, Math.floor(y / scale));
    for (let x = 0; x < width; x++) {
      const sourceX = Math.min(source.width - 1, Math.floor(x / scale));
      const sourceOffset = (sourceY * source.width + sourceX) * 4;
      const targetOffset = (y * width + x) * 4;
      data[targetOffset] = source.data[sourceOffset]!;
      data[targetOffset + 1] = source.data[sourceOffset + 1]!;
      data[targetOffset + 2] = source.data[sourceOffset + 2]!;
      data[targetOffset + 3] = source.data[sourceOffset + 3]!;
    }
  }

  return { width, height, data };
}

/** Detects the inner letter lattice at a bounded resolution, then returns
 * coordinates in the caller's original image space. */
export function detectBoardQuad(source: PixelBuffer, options: DetectQuadOptions = {}): [Point, Point, Point, Point] | null {
  const scale = Math.min(1, MAX_DETECTION_DIMENSION / Math.max(source.width, source.height));
  if (scale === 1) return detectBoardQuadAtWorkingResolution(source, options);

  const quad = detectBoardQuadAtWorkingResolution(downsampleForDetection(source, scale), options);
  return quad?.map((point) => ({ x: point.x / scale, y: point.y / scale })) as [Point, Point, Point, Point] | null;
}
