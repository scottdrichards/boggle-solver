import type { BackendName } from "../ml/backendChain";
import type { Point } from "../cv/quadWarp";

/** Structurally identical to `PixelBuffer` (`cv/quadWarp.ts`) — named
 * separately here only because crossing the worker boundary is what makes it
 * interesting: `data.buffer` travels in the transfer list, not by copy. */
export interface TransferablePixelBuffer {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

export interface InitMessage {
  type: "init";
  /** Resolved on the main thread via `backend.ts`'s `requestedBackend()` —
   * the worker cannot read `location`/`localStorage` itself, see that
   * function's doc comment. `null` means no override: run the normal
   * best-available chain. */
  backendOverride: BackendName | null;
}

/** One video frame, captured via `createImageBitmap(video)` at the stream's
 * native resolution — no resize on the main thread. Deliberately cheap:
 * creating a bitmap is compositor work, not a CPU pixel readback, so this
 * replaces what used to be a synchronous `drawImage`+`getImageData` on the
 * main thread on *every* frame. The worker performs both the cheap
 * detect-resolution draw and, only when a board is actually found, the
 * larger working-resolution draw from this same bitmap — one message in,
 * still only paying for the big readback when it's earned. */
export interface FrameMessage {
  type: "frame";
  requestId: number;
  bitmap: ImageBitmap;
  /** Spend the full coarse-to-fine pyramid instead of one quick look — see
   * `gridDetector.ts`. */
  deep: boolean;
}

export type PipelineRequest = InitMessage | FrameMessage;

export interface ReadyMessage {
  type: "ready";
  backend: BackendName;
}

export interface LoadErrorMessage {
  type: "load-error";
  message: string;
}

export interface FitSummary {
  /** In the *detect*-resolution frame's pixel space (the bitmap as the
   * worker first drew it down for detection) — matches what `BoardLockParams`
   * expects (`boardView.ts`), which only ever wanted this for computing an
   * on-screen scale factor, never the pixel data itself. */
  readonly quad: readonly [Point, Point, Point, Point];
  readonly gridSize: number;
  readonly inlierCount: number;
  readonly fill: number;
}

export interface PipelineFramePhases {
  /** Drawing the bitmap into the small detect-resolution canvas + reading it
   * back — the worker-side equivalent of the main thread's old `captureMs`
   * (see `telemetry.ts`), just relocated. Does *not* include the larger
   * working-resolution draw that only happens on a found board — that one
   * was never separately timed before this migration either; it still falls
   * into the "unaccounted" bucket `totalMs` minus the sum of phases exists
   * to surface. */
  captureMs: number;
  detectMs: number;
  detectSampleMs: number;
  detectPredictMs: number;
  detectFitMs: number;
  inliers: number;
  dispatches: number;
  passes: number;
  warpMs: number;
  sliceMs: number;
  classifyMs: number;
}

export interface PredictionSummary {
  readonly label: string;
  readonly confidence: number;
}

/** Detection ran but found nothing confident enough — the frame-miss case.
 * Still carries `phases` (a partial fit's inlier count included) because the
 * on-screen diagnostics and telemetry want it either way. */
export interface MissMessage {
  type: "miss";
  requestId: number;
  phases: PipelineFramePhases;
  backend: BackendName;
  tensors: number;
  tensorMB: number;
  /** Dimensions of the detect-resolution draw, in source (bitmap) pixels —
   * `BoardLockParams.frameWidth/frameHeight` need this even on a lock, and
   * the scanner needs it every frame for consistency; sent on every reply
   * rather than requiring the caller to also know the resize formula. */
  detectWidth: number;
  detectHeight: number;
}

export interface ResultMessage {
  type: "result";
  requestId: number;
  fit: FitSummary;
  detectWidth: number;
  detectHeight: number;
  /** Dimensions of the working-resolution draw used for warping — what
   * `telemetry.ts`'s `ScanReport.workingResolution` reports; unlike
   * `detectWidth/Height` this only exists once a board is confident enough
   * to warp, hence its own fields rather than reusing `detectWidth/Height`. */
  fullWidth: number;
  fullHeight: number;
  warped: TransferablePixelBuffer;
  cells: readonly TransferablePixelBuffer[];
  predictions: readonly PredictionSummary[];
  phases: PipelineFramePhases;
  backend: BackendName;
  tensors: number;
  tensorMB: number;
}

export interface FrameErrorMessage {
  type: "frame-error";
  requestId: number;
  phase: "detect" | "classify";
  message: string;
}

export type PipelineResponse = ReadyMessage | LoadErrorMessage | MissMessage | ResultMessage | FrameErrorMessage;
