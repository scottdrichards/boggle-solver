/// <reference lib="webworker" />
/**
 * CV/ML pipeline worker: board detection + letter classification, off the
 * main thread.
 *
 * Everything that used to run inline in `scanner.ts`'s frame loop —
 * `sampleRegion`/`fitLattice` (detection math), `warpQuadToSquare`/
 * `sliceIntoCells` (perspective correction), and both tfjs models — now runs
 * here instead. All of it is pure CPU/GPU work with no DOM dependency beyond
 * `OffscreenCanvas` (for turning an `ImageBitmap` into pixel data), so every
 * module imported below is the exact same one the main thread used before —
 * nothing was rewritten, just relocated.
 *
 * Why this is worth doing: on a *successful* frame, the main thread used to
 * pay for lattice fitting, a 640x640 perspective warp and GPU dispatch/
 * readback synchronously, in the middle of a loop that also owns the camera
 * preview and the UI. That is tens of ms the compositor cannot paint during,
 * every time the scanner finds something — and if a GPU backend call ever
 * truly hangs (the open, unconfirmed "page went unresponsive after
 * backgrounding" report — see this app's CLAUDE.md), it hangs the *entire
 * page*, input included, with no way to cancel it. Here, a hang is scoped to
 * this worker: `pipelineClient.ts` can `.terminate()` it outright and start
 * a fresh one, which is a real cancellation instead of "abandon a promise
 * that can't be cancelled".
 *
 * One `ImageBitmap` comes in per frame (`pipelineProtocol.ts`'s `FrameMessage`
 * — see its doc comment for why a bitmap, not a pre-drawn buffer). This
 * worker draws it down twice, at most: once at `DETECT_MAX_DIMENSION` for
 * every frame (cheap, matches what detection actually needs), and — only if
 * that pass finds a confident board — again at `WORKING_MAX_DIMENSION` to
 * warp and classify from. Same two-tier trade as before, same reasoning,
 * just executed here instead of on the main thread.
 */
import { CONFIDENT_FILL, type GridDetection } from "../cv/gridDetector";
import { sliceIntoCells, warpQuadToSquare, type PixelBuffer, type Point } from "../cv/quadWarp";
import { getClassifier, getDetector, setAssetBase as setModelAssetBase } from "../ml/models";
import {
  activeBackend,
  ensureBackend,
  setAssetBase as setWasmAssetBase,
  tensorMemory,
  type BackendName,
} from "../ml/backend";
import type {
  FrameErrorMessage,
  FrameMessage,
  PipelineFramePhases,
  MissMessage,
  PipelineRequest,
  PipelineResponse,
  ResultMessage,
  TransferablePixelBuffer,
} from "./pipelineProtocol";

/** See `scanner.ts`'s former `captureInto` doc comment for the origin of
 * this split — unchanged rationale, just moved here with the code that now
 * acts on it. */
const DETECT_MAX_DIMENSION = 640;
const WORKING_MAX_DIMENSION = 1400;
const WARP_OUTPUT_SIZE = 640;

/** Largest-first preference, tie-broken toward the smaller board inside
 * `fitLattice`: a 4x4 board's 16 peaks can't win a 5x5 reading unless they
 * actually explain more sites, so it's safe to try both every frame rather
 * than requiring the user to pick a size up front. */
const GRID_SIZES = [5, 4];

function post(message: PipelineResponse, transfer?: Transferable[]): void {
  self.postMessage(message, { transfer: transfer ?? [] });
}

/** Cached per size class so resizing (and therefore reallocating the backing
 * store) only happens when the video's actual resolution changes — same
 * reasoning as the main-thread canvas-reuse fix documented in this app's
 * CLAUDE.md ("Leaving the camera up for a while crashed the tab"). */
interface CanvasSlot {
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
}

function makeSlot(): CanvasSlot {
  const canvas = new OffscreenCanvas(1, 1);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return { canvas, ctx };
}

const detectSlot = makeSlot();
const fullSlot = makeSlot();

/** Draws `bitmap` into `slot`'s canvas at `maxDimension`, downscaled
 * proportionally, and reads back the pixels. Does not close `bitmap` — the
 * caller may draw it again into the other slot first. */
function drawToBuffer(bitmap: ImageBitmap, slot: CanvasSlot, maxDimension: number): PixelBuffer {
  const scale = Math.min(1, maxDimension / Math.max(bitmap.width, bitmap.height));
  const targetWidth = Math.round(bitmap.width * scale);
  const targetHeight = Math.round(bitmap.height * scale);

  if (slot.canvas.width !== targetWidth || slot.canvas.height !== targetHeight) {
    slot.canvas.width = targetWidth;
    slot.canvas.height = targetHeight;
    slot.ctx.imageSmoothingEnabled = true;
    slot.ctx.imageSmoothingQuality = "high";
  }
  slot.ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
  const imageData = slot.ctx.getImageData(0, 0, targetWidth, targetHeight);
  return { width: imageData.width, height: imageData.height, data: imageData.data };
}

function toTransferable(buffer: PixelBuffer): TransferablePixelBuffer {
  return { width: buffer.width, height: buffer.height, data: buffer.data };
}

async function handleInit(backendOverride: BackendName | null, assetBase: string): Promise<void> {
  try {
    // Must happen before ensureBackend()/getDetector()/getClassifier() ever
    // touch a URL — both default to a relative path that's wrong inside a
    // worker. See ml/models.ts's `assetBase` doc comment.
    setModelAssetBase(assetBase);
    setWasmAssetBase(assetBase);
    await ensureBackend(backendOverride);
    // Load both up front rather than lazily on first frame: the whole point
    // of the main thread calling this the moment the scanner opens (mirrors
    // the old `prefetchModels()` call site) is to hide the download/shader-
    // compile latency behind the camera permission prompt.
    await Promise.all([getDetector(), getClassifier()]);
    post({ type: "ready", backend: activeBackend() ?? "cpu" });
  } catch (error) {
    post({ type: "load-error", message: String(error) });
  }
}

async function handleFrame(message: FrameMessage): Promise<void> {
  const { requestId, bitmap, deep } = message;

  // Two separate try/catches, matching the two independent failure sites
  // scanner.ts used to guard separately: a detect failure and a classify
  // failure are diagnosed differently, so the `phase` reported to the main
  // thread has to reflect which one actually happened, not always "classify".
  let detectBuffer: PixelBuffer;
  let fit: GridDetection | null;
  let detectPhasesBase: Omit<PipelineFramePhases, "warpMs" | "sliceMs" | "classifyMs">;
  try {
    const captureStart = performance.now();
    detectBuffer = drawToBuffer(bitmap, detectSlot, DETECT_MAX_DIMENSION);
    const captureMs = performance.now() - captureStart;
    const detector = await getDetector();

    const detectStart = performance.now();
    fit = await detector.detect(detectBuffer, {
      gridSizes: GRID_SIZES,
      // `scales: [1]` = the centred-square first look plus square crops
      // covering the frame, all undistorted — must NOT be `scales: []`, see
      // gridDetector.ts's header comment.
      ...(deep ? {} : { scales: [1], refine: true }),
    });
    detectPhasesBase = {
      captureMs,
      detectMs: performance.now() - detectStart,
      detectSampleMs: fit?.timings.sampleMs ?? 0,
      detectPredictMs: fit?.timings.predictMs ?? 0,
      detectFitMs: fit?.timings.fitMs ?? 0,
      inliers: fit?.inlierCount ?? 0,
      dispatches: fit?.batches ?? 0,
      passes: fit?.passes ?? 0,
    };
  } catch (error) {
    try {
      bitmap.close();
    } catch {
      /* already closed, or never fully opened — either way nothing to free */
    }
    const errorMessage: FrameErrorMessage = { type: "frame-error", requestId, phase: "detect", message: String(error) };
    post(errorMessage);
    return;
  }

  const stats = tensorMemory();
  const backend = activeBackend() ?? "cpu";

  if (!fit || fit.fill < CONFIDENT_FILL) {
    bitmap.close();
    const missPhases: PipelineFramePhases = { ...detectPhasesBase, warpMs: 0, sliceMs: 0, classifyMs: 0 };
    const miss: MissMessage = {
      type: "miss",
      requestId,
      phases: missPhases,
      backend,
      tensors: stats.numTensors,
      tensorMB: stats.numBytes / 1e6,
      detectWidth: detectBuffer.width,
      detectHeight: detectBuffer.height,
    };
    post(miss);
    return;
  }

  try {
    // Board found: now it's worth paying for detail. Re-draw the same
    // bitmap at full working resolution and scale the quad into that space.
    const fullBuffer = drawToBuffer(bitmap, fullSlot, WORKING_MAX_DIMENSION);
    bitmap.close();
    const ratio = fullBuffer.width / detectBuffer.width;
    const quad = fit.quad.map((point: Point) => ({ x: point.x * ratio, y: point.y * ratio })) as [
      Point,
      Point,
      Point,
      Point,
    ];

    const warpStart = performance.now();
    const warped = warpQuadToSquare(fullBuffer, quad, WARP_OUTPUT_SIZE);
    const warpMs = performance.now() - warpStart;

    const sliceStart = performance.now();
    // Sliced by the size this frame's fit actually found, not any size the
    // caller assumed — the fitter is a fresh guess every frame.
    const cells = sliceIntoCells(warped, fit.gridSize);
    const sliceMs = performance.now() - sliceStart;

    const classifyStart = performance.now();
    const classifier = await getClassifier();
    const predictions = await classifier.classifyCells(cells);
    const classifyMs = performance.now() - classifyStart;

    const finalStats = tensorMemory();
    const result: ResultMessage = {
      type: "result",
      requestId,
      fit: { quad: fit.quad, gridSize: fit.gridSize, inlierCount: fit.inlierCount, fill: fit.fill },
      detectWidth: detectBuffer.width,
      detectHeight: detectBuffer.height,
      fullWidth: fullBuffer.width,
      fullHeight: fullBuffer.height,
      warped: toTransferable(warped),
      cells: cells.map(toTransferable),
      predictions,
      phases: { ...detectPhasesBase, warpMs, sliceMs, classifyMs },
      backend: activeBackend() ?? "cpu",
      tensors: finalStats.numTensors,
      tensorMB: finalStats.numBytes / 1e6,
    };
    post(result, [warped.data.buffer, ...cells.map((c) => c.data.buffer)]);
  } catch (error) {
    // `bitmap.close()` already ran above before this try started, so nothing
    // to free here — just report which phase failed.
    const errorMessage: FrameErrorMessage = {
      type: "frame-error",
      requestId,
      phase: "classify",
      message: String(error),
    };
    post(errorMessage);
  }
}

self.addEventListener("message", (event: MessageEvent<PipelineRequest>) => {
  const message = event.data;
  if (message.type === "init") {
    void handleInit(message.backendOverride, message.assetBase);
  } else if (message.type === "frame") {
    void handleFrame(message);
  }
});
