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
  letters?: string[];
}

const ENDPOINT = `${import.meta.env.BASE_URL}api/telemetry`;

function newSessionId(): string {
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export class ScanTelemetry {
  readonly sessionId = newSessionId();
  private frames: FramePhases[] = [];
  private startedAt = 0;

  begin(): void {
    this.frames = [];
    this.startedAt = performance.now();
  }

  record(frame: FramePhases): void {
    this.frames.push(frame);
  }

  /** Sends and clears. Never throws and never blocks the caller. */
  send(report: Omit<ScanReport, "sessionId" | "frames" | "totalMs">, letters?: string[]): void {
    const payload: ScanReport = {
      ...report,
      sessionId: this.sessionId,
      totalMs: performance.now() - this.startedAt,
      frames: this.frames,
      letters,
    };
    const body = JSON.stringify(payload);

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
