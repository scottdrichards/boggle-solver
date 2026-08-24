/**
 * Per-frame scanner telemetry, POSTed to the verification server.
 *
 * The scanner's on-screen diagnostics show the *latest* frame, which is the
 * wrong shape for the question "why does a whole scan take 5 seconds". That
 * needs every frame's phase breakdown plus the wall-clock between them —
 * including the gaps that belong to no phase at all, which is exactly where a
 * surprise is likely to be hiding.
 *
 * Fire-and-forget by design: this is a diagnostic aid, and a scan must never
 * get slower or fail because a telemetry POST did. On GitHub Pages there is no
 * endpoint at all and every send simply no-ops.
 */

export interface FramePhases {
  /** Grabbing the video frame into the offscreen canvas + getImageData. */
  captureMs: number;
  /** Whole detect call, including the three sub-phases below. */
  detectMs: number;
  detectSampleMs: number;
  detectPredictMs: number;
  detectFitMs: number;
  /** Perspective warp to the 640 square. */
  warpMs: number;
  /** Slicing the square into 25 cells. */
  sliceMs: number;
  /** Classifier, including the 4-rotation TTA. */
  classifyMs: number;
  /** Wall-clock for the whole loop iteration, phases and gaps alike. The
   * difference between this and the sum of the phases is time the loop spent
   * outside anything we instrumented — yielding, GC, layout. */
  totalMs: number;
  deep: boolean;
  inliers: number;
  dispatches: number;
  passes: number;
  /** Cells still unsettled after this frame's vote. */
  unsettled: number;
  meanConfidence: number;
  /** Live tfjs tensors after this frame. Flat = no tensor leak; a steady
   * climb over a scan is a missing dispose/tidy. */
  tensors: number;
  /** tfjs-held bytes (GPU textures/buffers included), MB. */
  tensorMB: number;
  /** `performance.memory.usedJSHeapSize` in MB where the browser exposes it
   * (Chromium only), else 0. Deliberately separate from `tensorMB`: a scan
   * can die from either, and they have completely different fixes. */
  heapMB: number;
}

export interface ScanReport {
  sessionId: string;
  userAgent: string;
  backend: string;
  videoResolution: string;
  workingResolution: string;
  /** Page load -> scanner start, i.e. what the user waits through first. */
  startupMs: number;
  /** Scanner start -> lock (or give-up). The number the user actually feels. */
  totalMs: number;
  outcome: "locked" | "timeout" | "cancelled";
  frames: FramePhases[];
  /** Frames dropped from the head of `frames` because the scan outran
   * `MAX_FRAMES`. Non-zero means `frames` is the tail of a longer scan. */
  droppedFrames: number;
  letters?: string[];
}

/**
 * Hard cap on retained frames.
 *
 * This buffer used to be unbounded, which made it the one structure in the app
 * whose size grew with how long the camera was held up: a scan that never locks
 * records a frame every iteration forever. At a few frames a second that is
 * tens of thousands of objects — and then a multi-megabyte `JSON.stringify` at
 * the end, well past `sendBeacon`'s size limit, on a phone that is already
 * short of memory.
 *
 * The tail is what gets kept: a long scan's interesting part is the end (the
 * lock, or what it was doing when it gave up), and the head of a search that
 * ran for ten minutes is a thousand near-identical misses.
 */
const MAX_FRAMES = 600;

const ENDPOINT = `${import.meta.env.BASE_URL}api/telemetry`;

function newSessionId(): string {
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export class ScanTelemetry {
  readonly sessionId = newSessionId();
  private frames: FramePhases[] = [];
  private droppedFrames = 0;
  private startedAt = 0;

  begin(): void {
    this.frames = [];
    this.droppedFrames = 0;
    this.startedAt = performance.now();
  }

  record(frame: FramePhases): void {
    this.frames.push(frame);
    if (this.frames.length > MAX_FRAMES) {
      // Drop in one chunk rather than shifting every frame: a repeated
      // `shift()` on a 600-element array is per-frame work in the scan loop,
      // and this runs at most once per 10% of the cap.
      const excess = this.frames.length - MAX_FRAMES;
      this.frames.splice(0, excess + MAX_FRAMES / 10);
      this.droppedFrames += excess + MAX_FRAMES / 10;
    }
  }

  /** Sends and clears. Never throws and never blocks the caller. */
  send(report: Omit<ScanReport, "sessionId" | "frames" | "droppedFrames" | "totalMs">, letters?: string[]): void {
    const payload: ScanReport = {
      ...report,
      sessionId: this.sessionId,
      totalMs: performance.now() - this.startedAt,
      frames: this.frames,
      droppedFrames: this.droppedFrames,
      letters,
    };
    const body = JSON.stringify(payload);
    // Actually clear, as the name promises. `begin()` used to be the only
    // thing that emptied the buffer, so a send with no restart after it (a
    // lock that is never followed by "scan a new board") left every frame of
    // that scan retained for the rest of the page's life.
    this.frames = [];
    this.droppedFrames = 0;

    // sendBeacon survives the page being backgrounded mid-scan, which fetch
    // does not reliably do; fall back where it is unavailable or refuses.
    try {
      if (navigator.sendBeacon?.(ENDPOINT, new Blob([body], { type: "application/json" }))) return;
    } catch {
      // fall through to fetch
    }
    void fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: true,
    }).catch(() => {
      // No endpoint (e.g. GitHub Pages). Diagnostics are optional.
    });
  }
}
