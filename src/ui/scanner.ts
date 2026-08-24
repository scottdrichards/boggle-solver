/**
 * Live scanner: point the phone at the board and it reads itself.
 *
 * The flow is a loop, not a wizard — open the camera, keep looking, lock when
 * the board has been read the same way several times running, then *keep
 * looking* so a new board placed in view gets picked up on its own. There is
 * no shutter, no corner dragging, no manual fallback path: the camera and the
 * word list are the entire app.
 *
 * Three things about the design are load-bearing:
 *
 * 1. **Frames are cheap looks, so spend them on agreement rather than on one
 *    careful analysis.** Each frame runs a cheap square-crop detector pass
 *    rather than the full coarse-to-fine pyramid; the pyramid is kept only as
 *    a periodic fallback for when the quick look keeps missing.
 *
 * 2. **The overlay never touches the pixels the models see.** The preview and
 *    its detection overlay are separate elements from the offscreen canvas
 *    frames are captured into. This is structural, not incidental: painting
 *    the cell-centre dots onto the same canvas the crops were cut from once
 *    cost 15 points of end-to-end accuracy (see the project brief), and the
 *    only reliable fix is for the drawn-on surface not to exist in the
 *    capture path at all.
 *
 * 3. **A lock does not stop the scan.** The camera stays live and the loop
 *    keeps voting; `onLocked` only fires again once the settled letters
 *    actually change, so holding the camera on a solved board is silent and
 *    swapping in a new one just works.
 */
import { CONFIDENT_INLIERS } from "../cv/gridDetector";
import { sliceIntoCells, warpQuadToSquare, type PixelBuffer, type Point } from "../cv/quadWarp";
import { getClassifier, getDetector, prefetchModels } from "../ml/models";
import type { Prediction } from "../ml/classifier";
import { activeBackend, tensorMemory } from "../ml/backend";
import { ScanConsensus, type LockedReading } from "./consensus";
import { ScanTelemetry, type FramePhases } from "./telemetry";
import { mountBoardView, LOCK_TRANSITION_MS } from "./boardView";
import { setTiles, state } from "./state";

/** Frames are captured at this working size, matching the photo path so a
 * crop cut from a frame is the same kind of image the model was tuned on. */
const WORKING_MAX_DIMENSION = 1400;
/** Same endpoint `serveTLS.py`/`submitServer.py` already implement — see
 * "Report wrong board" below. Relative + BASE_URL-prefixed so it resolves
 * correctly whether served from `/` or GitHub Pages' subpath; on GitHub
 * Pages there is no server behind it and the POST just fails, which the
 * report handler treats the same as any other network error. */
const SUBMIT_ENDPOINT = `${import.meta.env.BASE_URL}api/submit`;
/** Detection-only grab size. The detector resamples every crop to 128x128
 * regardless, so detail here is wasted — and this readback happens on every
 * frame, including the ones that find nothing. */
const DETECT_MAX_DIMENSION = 640;
const WARP_OUTPUT_SIZE = 640;
/** Voting window. See consensus.ts: cells settle individually, so a single
 * flickering die cannot veto a board the other 24 cells agree on. */
const CONSENSUS_OPTIONS = {
  windowSize: 6,
  minFrames: 3,
  minAgreementRatio: 0.6,
  minMeanConfidence: 0.55,
  minFrameConfidence: 0.4,
} as const;
/** After this many quick looks in a row find nothing, spend one full pyramid
 * scan — the board may be further away than the single-pass range covers. */
const QUICK_LOOKS_BEFORE_PYRAMID = 6;

/** Reported symptom (2026-08-24, Android Chrome/Edge, deployed build): the
 * whole page goes unresponsive after backgrounding the tab and coming back.
 * `nextFrame`'s own timeout only covers one rAF wait, not an `await` that
 * never settles at all — e.g. `getUserMedia` re-acquiring a camera the OS
 * hasn't actually released yet, or a WebGPU device request stuck after the
 * tab's GPU context was reclaimed while hidden. Neither has been reproduced
 * (no browser on the agent box), so this is a recovery net, not a fix for a
 * confirmed root cause: if no frame makes progress for this long while the
 * loop believes it's running, force-cycle the camera rather than sit stalled
 * forever. Checked well below the interval so one slow deep-pyramid frame
 * never trips it. */
const WATCHDOG_STALL_MS = 15000;
const WATCHDOG_CHECK_MS = 4000;
/** After this many forced recoveries in one page life, stop retrying and
 * leave the error panel up — a device that can't hold a camera stream for
 * more than a few seconds needs a human, not another silent retry loop. */
const MAX_AUTO_RECOVERIES = 3;

/** Breadcrumb for a freeze/crash with no console to read: written whenever
 * the camera is live, cleared only on a clean `pagehide`. If a future launch
 * finds a stale entry, the previous session ended without that handler
 * firing — evidence for exactly the "whole page unresponsive" report this
 * file otherwise has none for. */
const HEARTBEAT_KEY = "boggle.scanner.heartbeat";

function writeHeartbeat(extra: Record<string, unknown> = {}): void {
  try {
    localStorage.setItem(
      HEARTBEAT_KEY,
      JSON.stringify({ at: Date.now(), ua: navigator.userAgent, ...extra }),
    );
  } catch {
    // Private mode / storage disabled — the breadcrumb is best-effort only.
  }
}

function clearHeartbeat(): void {
  try {
    localStorage.removeItem(HEARTBEAT_KEY);
  } catch {
    /* see writeHeartbeat */
  }
}

/** Reads and clears any breadcrumb left by a previous session that never hit
 * `pagehide` — call once at mount, before the first `writeHeartbeat`. */
function readStaleHeartbeat(): Record<string, unknown> | null {
  try {
    const raw = localStorage.getItem(HEARTBEAT_KEY);
    if (!raw) return null;
    localStorage.removeItem(HEARTBEAT_KEY);
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export interface ScannerHandlers {
  /** The settled letters changed — either the first lock, or a new board was
   * swapped in. The scan keeps running underneath this. */
  onLocked(): void;
}

export interface ScannerController {
  /** Opens the camera and starts scanning. Safe to call when already open. */
  start(): Promise<void>;
  stop(): void;
  /** Highlights a word's path (row-major cell indices, in visiting order) on
   * the frozen board, or clears it when `null`. No-op before a lock. */
  showPath(path: readonly number[] | null): void;
  /** Paints a per-cell point heatmap on the frozen board, or clears it when
   * `null` — see `BoardView.showHeatmap`. No-op before a lock. */
  showHeatmap(weights: ArrayLike<number> | null): void;
  /** Rapid "thinking" flourish through several words' paths — see
   * `BoardView.playFlourish`. No-op before a lock. */
  playFlourish(paths: readonly (readonly number[])[], totalMs: number): void;
}

interface FrameCapture {
  buffer: PixelBuffer;
  canvas: HTMLCanvasElement;
}

export function mountScanner(container: HTMLElement, handlers: ScannerHandlers): ScannerController {
  container.innerHTML = `
    <div class="scanner-stage" id="scanner-stage">
      <video id="scanner-video" playsinline muted autoplay></video>
      <canvas id="scanner-overlay"></canvas>
      <div class="scanner-status" id="scanner-status">Starting camera…</div>
    </div>
    <p class="scanner-error" id="scanner-error" hidden></p>
    <div class="scanner-actions">
      <button type="button" id="scanner-retry" hidden>Retry camera</button>
      <button type="button" id="scanner-review" hidden>Review letters</button>
      <button type="button" id="scanner-report" hidden>Report wrong board</button>
      <button type="button" id="scanner-rescan" hidden>Scan a new board</button>
    </div>
    <pre class="scanner-diagnostics" id="scanner-diagnostics" hidden></pre>
  `;

  const stage = container.querySelector<HTMLElement>("#scanner-stage")!;
  const video = container.querySelector<HTMLVideoElement>("#scanner-video")!;
  const overlay = container.querySelector<HTMLCanvasElement>("#scanner-overlay")!;
  const statusEl = container.querySelector<HTMLElement>("#scanner-status")!;
  const errorEl = container.querySelector<HTMLElement>("#scanner-error")!;
  const retryBtn = container.querySelector<HTMLButtonElement>("#scanner-retry")!;
  const reviewBtn = container.querySelector<HTMLButtonElement>("#scanner-review")!;
  const reportBtn = container.querySelector<HTMLButtonElement>("#scanner-report")!;
  const rescanBtn = container.querySelector<HTMLButtonElement>("#scanner-rescan")!;
  const diagnosticsEl = container.querySelector<HTMLElement>("#scanner-diagnostics")!;
  const overlayCtx = overlay.getContext("2d")!;
  // Presentation after a lock: freezing the frame, animating the quad into a
  // flat square, and drawing a tapped word's trail on it. See boardView.ts.
  const boardView = mountBoardView(stage, video);

  // The board starts full-size so it's easy to read while scanning, but once
  // the player scrolls into the results list it shrinks and sticks to the
  // top (see .scanner-panel in style.css) so the board stays visible as a
  // reference without eating the space the word list needs. It only shrinks
  // in response to an actual scroll — not merely "after a lock" — so a board
  // still being read at the top of the page stays full-size.
  //
  // Sizing is deliberately computed and applied here in JS, as an explicit
  // width+height pair that always keeps STAGE_ASPECT, rather than done in
  // CSS via width:100%/aspect-ratio/max-height. Two earlier CSS attempts
  // both broke: transitioning max-height and max-width together fought
  // aspect-ratio's own derivation and looked jittery, and transitioning only
  // max-height left width at 100%, so the box's own aspect ratio changed —
  // stretching the (square) flattened-board canvas into a squished
  // wide/short rectangle instead of shrinking it proportionally. Since both
  // the natural and compact sizes are computed with the same ratio and set
  // as concrete pixel values, the CSS transition just interpolates two
  // definite numbers — always proportional, never squished.
  const STAGE_ASPECT = 3 / 4; // width:height, matches the video's own framing
  const NATURAL_MAX_HEIGHT_VH = 70;
  const COMPACT_HEIGHT_VH = 22;

  function computeStageSize(targetHeightVh: number): { width: number; height: number } {
    const columnWidth = container.clientWidth || stage.clientWidth || 1;
    const heightCapPx = (targetHeightVh / 100) * window.innerHeight;
    const height = Math.min(columnWidth / STAGE_ASPECT, heightCapPx);
    return { width: height * STAGE_ASPECT, height };
  }

  const COMPACT_ENTER_PX = 56;
  const COMPACT_EXIT_PX = 12;
  let compact = false;
  let compactUpdateScheduled = false;

  function applyStageSize(): void {
    const { width, height } = computeStageSize(compact ? COMPACT_HEIGHT_VH : NATURAL_MAX_HEIGHT_VH);
    stage.style.width = `${width}px`;
    stage.style.height = `${height}px`;
  }

  function updateCompactState(): void {
    compactUpdateScheduled = false;
    const wasCompact = compact;
    if (!compact && window.scrollY > COMPACT_ENTER_PX) compact = true;
    else if (compact && window.scrollY < COMPACT_EXIT_PX) compact = false;
    container.classList.toggle("scanner-panel--compact", compact);
    if (compact !== wasCompact) applyStageSize();
  }
  window.addEventListener(
    "scroll",
    () => {
      // Coalesce to at most one check per animation frame — scroll fires far
      // more often than that, and this avoids re-triggering layout mid-scroll.
      if (compactUpdateScheduled) return;
      compactUpdateScheduled = true;
      requestAnimationFrame(updateCompactState);
    },
    { passive: true },
  );
  window.addEventListener("resize", applyStageSize, { passive: true });
  applyStageSize();
  updateCompactState();

  // The capture surface. Never displayed, never drawn on by the UI — see the
  // note at the top of this file about why that separation is structural.
  const frameCanvas = document.createElement("canvas");
  const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true })!;
  frameCtx.imageSmoothingEnabled = true;
  frameCtx.imageSmoothingQuality = "high";
  /** Small buffer used for detection only — see `captureInto`. */
  const detectCanvas = document.createElement("canvas");
  const detectCtx = detectCanvas.getContext("2d", { willReadFrequently: true })!;
  detectCtx.imageSmoothingEnabled = true;
  detectCtx.imageSmoothingQuality = "high";

  let stream: MediaStream | null = null;
  let running = false;
  /** Whether the loop should be live once permission/visibility allow it —
   * distinct from `running`, which tracks the camera itself. A backgrounded
   * tab pauses the camera without the user ever having asked to stop. */
  let wantsToRun = false;
  /** In-flight `start()`, so two callers cannot each open a camera stream.
   * `running` alone could not do this: it is only set after `getUserMedia`
   * resolves, so two rapid starts both sailed past the guard and the first
   * stream was leaked — the camera stayed live with nothing reading it. */
  let starting: Promise<void> | null = null;
  /** Deferred camera release scheduled by `lockIn` — see there. */
  let pendingStop: number | null = null;
  let loopToken = 0;
  let quickLookMisses = 0;
  let startupMs = 0;
  /** Watchdog bookkeeping — see WATCHDOG_STALL_MS above. `lastProgressAt`
   * covers a stuck loop iteration; `startInitiatedAt` covers a `startInner()`
   * that never gets as far as `running = true` at all (a `getUserMedia` that
   * never settles, e.g. re-acquiring a camera the OS hasn't released yet). */
  let lastProgressAt = performance.now();
  let startInitiatedAt: number | null = null;
  let autoRecoveries = 0;

  {
    const stale = readStaleHeartbeat();
    if (stale) {
      // "backgrounded" is expected (the OS reclaimed a hidden tab); anything
      // else means the previous session was actively holding the camera and
      // never reached pagehide — the closest thing to evidence this file has
      // for the "whole page unresponsive" report with no console to read.
      console.warn("scanner: previous session did not exit cleanly", stale);
    }
  }
  const consensus = new ScanConsensus(CONSENSUS_OPTIONS);
  const telemetry = new ScanTelemetry();

  /** Snapshot of the current lock, kept only so "Report wrong board" has
   * something to send — there is no manual-correction UI any more (see the
   * project brief), so this reports the board as-read rather than
   * as-corrected. A human sorts flagged-vs-fine submissions out later from
   * `meta.json`'s `reportedWrong` field, same pipeline the removed capture.ts
   * fed via `/api/submit`. */
  let pendingReport: {
    photo: HTMLCanvasElement;
    letters: string[];
    confidences: (number | null)[];
    quad: [Point, Point, Point, Point];
    frameWidth: number;
    frameHeight: number;
  } | null = null;

  /** On-screen diagnostics, opt-in via `?debug=1`. There is no console on a
   * phone, and these were how the aspect-distortion bug got found — but the
   * full breakdown now goes to telemetry on every scan, so the shipped UI
   * does not need to carry it. */
  const showDiagnostics = new URLSearchParams(location.search).get("debug") === "1";
  diagnosticsEl.hidden = !showDiagnostics;

  const diagnostics = {
    backend: "?",
    resolution: "?",
    detectMs: 0,
    classifyMs: 0,
    inliers: 0,
    dispatches: 0,
    deep: false,
    frames: 0,
    unsettled: 0,
    meanConfidence: 0,
    tensors: 0,
    tensorMB: 0,
    heapMB: 0,
    note: "",
  };

  function renderDiagnostics(): void {
    if (!showDiagnostics) return;
    diagnosticsEl.textContent = [
      `backend  ${diagnostics.backend}   video ${diagnostics.resolution}`,
      `detect   ${diagnostics.detectMs.toFixed(0)}ms  (${diagnostics.dispatches} dispatch${
        diagnostics.dispatches === 1 ? "" : "es"
      }${diagnostics.deep ? ", deep" : ""})  inliers ${diagnostics.inliers}/25`,
      `classify ${diagnostics.classifyMs.toFixed(0)}ms`,
      `votes    ${diagnostics.frames} frames, ${diagnostics.unsettled} cells unsettled, conf ${diagnostics.meanConfidence.toFixed(
        2,
      )}`,
      `memory   ${diagnostics.tensors} tensors, ${diagnostics.tensorMB.toFixed(1)}MB tfjs${
        diagnostics.heapMB > 0 ? `, ${diagnostics.heapMB.toFixed(0)}MB heap` : ""
      }`,
      diagnostics.note,
    ]
      .filter(Boolean)
      .join("\n");
  }

  /** Folds one loop iteration into the telemetry buffer. Misses are recorded
   * too: a scan that feels slow may be spending its time failing to find the
   * board, which looks nothing like a scan that is slow to classify. */
  function recordFrame(
    phases: Partial<FramePhases>,
    deep: boolean,
    iterationStart: number,
    progress: { unsettled: number; meanConfidence: number },
  ): void {
    const tensors = tensorMemory();
    // Non-standard and Chromium-only; absent everywhere else, hence the cast
    // rather than a global type declaration for something that may not exist.
    const heapBytes = (performance as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? 0;
    diagnostics.tensors = tensors.numTensors;
    diagnostics.tensorMB = tensors.numBytes / 1e6;
    diagnostics.heapMB = heapBytes / 1e6;
    telemetry.record({
      captureMs: phases.captureMs ?? 0,
      detectMs: phases.detectMs ?? 0,
      detectSampleMs: phases.detectSampleMs ?? 0,
      detectPredictMs: phases.detectPredictMs ?? 0,
      detectFitMs: phases.detectFitMs ?? 0,
      warpMs: phases.warpMs ?? 0,
      sliceMs: phases.sliceMs ?? 0,
      classifyMs: phases.classifyMs ?? 0,
      totalMs: performance.now() - iterationStart,
      deep,
      inliers: phases.inliers ?? 0,
      dispatches: phases.dispatches ?? 0,
      passes: phases.passes ?? 0,
      unsettled: progress.unsettled,
      meanConfidence: progress.meanConfidence,
      tensors: tensors.numTensors,
      tensorMB: tensors.numBytes / 1e6,
      heapMB: heapBytes / 1e6,
    });
  }

  function sendTelemetry(outcome: "locked" | "cancelled", letters?: string[]): void {
    telemetry.send(
      {
        userAgent: navigator.userAgent,
        backend: activeBackend() ?? "unknown",
        videoResolution: `${video.videoWidth}x${video.videoHeight}`,
        workingResolution: `${frameCanvas.width}x${frameCanvas.height}`,
        startupMs: startupMs,
        outcome,
      },
      letters,
    );
  }

  function setStatus(text: string): void {
    statusEl.textContent = text;
  }

  function showError(message: string, canRetry: boolean): void {
    errorEl.textContent = message;
    errorEl.hidden = false;
    retryBtn.hidden = !canRetry;
  }

  function clearError(): void {
    errorEl.hidden = true;
    retryBtn.hidden = true;
  }

  /** The detection overlay used to draw a yellow dot on every fitted cell
   * centre — removed: it was a debugging aid, but it drew a jarring blob on
   * top of every die each frame, over dice the player is trying to read.
   * The canvas element (and the object-fit-cover sizing math) stays, cleared
   * every frame, in case a future diagnostic wants a scratch overlay layer
   * again; nothing is drawn onto it today. */
  function drawOverlay(): void {
    const rect = video.getBoundingClientRect();
    if (overlay.width !== Math.round(rect.width) || overlay.height !== Math.round(rect.height)) {
      overlay.width = Math.round(rect.width);
      overlay.height = Math.round(rect.height);
    }
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  /** Grabs the video into `canvas` at `maxDimension` and reads it back.
   *
   * Resolution is split in two because the two consumers need wildly
   * different things. **Detection** downsamples every crop to 128x128 no
   * matter what it is given, so feeding it a 788x1400 buffer is pure waste —
   * and it is paid on every frame, including the ~93% that find nothing.
   * Measured on-device, this readback alone was 26-36% of a whole scan.
   * **Classification** is the opposite: its crops are cut straight from the
   * buffer, so it wants all the detail available.
   *
   * So the loop detects on a small grab and, only once a board is actually
   * found, pays for a full-resolution one to warp from. */
  function captureInto(
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    maxDimension: number,
  ): FrameCapture | null {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) return null;

    const scale = Math.min(1, maxDimension / Math.max(width, height));
    const targetWidth = Math.round(width * scale);
    const targetHeight = Math.round(height * scale);
    // Only assign when the size actually changes. Assigning `canvas.width` is
    // not a no-op when the value is unchanged: it resets the drawing surface
    // and reallocates the backing store, which is native (often GPU) memory
    // the JS heap does not account for and the browser reclaims lazily. This
    // ran on *every* frame — ~0.9 MB for the detect grab plus ~4.4 MB for the
    // full-resolution one — so a camera left running churned tens of MB per
    // second of allocations nothing in the JS profile would show. Which is
    // exactly the shape of "left it up for a while and the tab died".
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      // Resizing resets context state, so the smoothing hints set at creation
      // have to be reapplied. They were silently being reset every frame
      // before, which is its own small bug.
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
    }
    // The video frame is opaque and covers the canvas exactly, so the stale
    // previous frame underneath is fully overwritten — no clear needed.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return { buffer: ctx.getImageData(0, 0, canvas.width, canvas.height), canvas };
  }

  /** One look at one frame. `deep` spends the full coarse-to-fine pyramid;
   * otherwise it is a single whole-frame pass, which is all a board held up
   * to the camera needs. */
  async function readFrame(frame: FrameCapture, deep: boolean, phases: Partial<FramePhases>) {
    const detector = await getDetector();

    const detectStart = performance.now();
    const fit = await detector.detect(frame.buffer, {
      gridSizes: [5],
      // `scales: [1]` = the centred-square first look plus square crops
      // covering the frame, all undistorted. It must NOT be `scales: []`:
      // that left only the first look, and before the centred-square fix
      // that look was an aspect-squashed whole frame which found the board
      // in 3 frames out of 42.
      ...(deep ? {} : { scales: [1], refine: true }),
    });
    phases.detectMs = performance.now() - detectStart;
    phases.detectSampleMs = fit?.timings.sampleMs ?? 0;
    phases.detectPredictMs = fit?.timings.predictMs ?? 0;
    phases.detectFitMs = fit?.timings.fitMs ?? 0;
    phases.inliers = fit?.inlierCount ?? 0;
    phases.dispatches = fit?.batches ?? 0;
    phases.passes = fit?.passes ?? 0;

    if (!fit || fit.inlierCount < CONFIDENT_INLIERS) return { fit: null, cells: null };

    // Board found, so now it is worth paying for detail: re-grab at full
    // working resolution and scale the quad into that larger space.
    const full = captureInto(frameCanvas, frameCtx, WORKING_MAX_DIMENSION) ?? frame;
    const ratio = full.buffer.width / frame.buffer.width;
    const quad = fit.quad.map((point) => ({ x: point.x * ratio, y: point.y * ratio })) as [
      Point,
      Point,
      Point,
      Point,
    ];

    const warpStart = performance.now();
    const warped = warpQuadToSquare(full.buffer, quad, WARP_OUTPUT_SIZE);
    phases.warpMs = performance.now() - warpStart;

    const sliceStart = performance.now();
    const cells = sliceIntoCells(warped, state.gridSize);
    phases.sliceMs = performance.now() - sliceStart;

    return { fit, cells, full, quad, warped };
  }

  /** Pauses the video (so its last frame stays visible under the flatten
   * animation) and stops the read loop immediately, but only releases the
   * camera hardware once the animation has had time to fully cover it —
   * releasing sooner would blank the video mid-transition, since clearing
   * `srcObject` drops the last frame instantly in most browsers. */
  function lockIn(
    locked: LockedReading,
    quadInFrame: [Point, Point, Point, Point],
    frame: FrameCapture,
    warped: PixelBuffer,
    cells: readonly PixelBuffer[],
    fullFrame: FrameCapture,
    fullQuad: [Point, Point, Point, Point],
  ): void {
    setTiles(locked.letters);
    sendTelemetry("locked", locked.letters);

    // Full working-resolution photo + the quad scaled to match it — same
    // pair `readFrame` used to warp/slice, so what gets submitted is exactly
    // what the classifier saw, not the smaller detect-only grab `frame` is.
    pendingReport = {
      photo: fullFrame.canvas,
      letters: [...locked.letters],
      confidences: [...locked.confidences],
      quad: fullQuad,
      frameWidth: fullFrame.buffer.width,
      frameHeight: fullFrame.buffer.height,
    };
    reportBtn.hidden = false;
    reportBtn.disabled = false;
    reportBtn.textContent = "Report wrong board";

    wantsToRun = false;
    running = false;
    loopToken++;
    video.pause();

    boardView.lock({
      quad: quadInFrame,
      frameWidth: frame.buffer.width,
      frameHeight: frame.buffer.height,
      board: warped,
      gridSize: state.gridSize,
      cells,
      letters: locked.letters,
      confidences: locked.confidences,
    });
    reviewBtn.hidden = false;
    reviewBtn.textContent = "Review letters";
    rescanBtn.hidden = false;
    setStatus("Board captured — tap a word to trace it, or scan a new board.");

    // Cancelled by `start()`: tapping "Scan a new board" during the flatten
    // animation would otherwise have this fire a moment later and stop the
    // camera that had just been reopened.
    pendingStop = window.setTimeout(() => {
      pendingStop = null;
      stop();
    }, LOCK_TRANSITION_MS + 100);

    handlers.onLocked();
  }

  async function loop(token: number): Promise<void> {
    while (running && token === loopToken) {
      lastProgressAt = performance.now();
      const iterationStart = performance.now();
      const phases: Partial<FramePhases> = {};

      const captureStart = performance.now();
      const frame = captureInto(detectCanvas, detectCtx, DETECT_MAX_DIMENSION);
      phases.captureMs = performance.now() - captureStart;
      if (!frame) {
        await nextFrame();
        continue;
      }

      const deep = quickLookMisses >= QUICK_LOOKS_BEFORE_PYRAMID;
      diagnostics.backend = activeBackend() ?? "loading";
      diagnostics.resolution = `${video.videoWidth}x${video.videoHeight}`;
      diagnostics.deep = deep;

      let result: Awaited<ReturnType<typeof readFrame>>;
      try {
        result = await readFrame(frame, deep, phases);
      } catch (error) {
        // A failure here is the interesting case and used to vanish into the
        // console, where a phone user could never see it.
        console.error("scanner frame failed", error);
        diagnostics.note = `detect failed: ${String(error).slice(0, 90)}`;
        renderDiagnostics();
        await nextFrame();
        continue;
      }
      diagnostics.detectMs = phases.detectMs ?? 0;
      diagnostics.inliers = phases.inliers ?? 0;
      diagnostics.dispatches = phases.dispatches ?? 0;
      if (token !== loopToken) return;

      if (!result.fit || !result.cells) {
        // A deep scan that found nothing goes back to the cheap cadence
        // rather than pinning every later frame to the pyramid: if the board
        // isn't there, it is not worth paying the expensive look every frame.
        quickLookMisses = deep ? 0 : quickLookMisses + 1;
        // Deliberately no `consensus.reset()`: one dropped frame — a hand
        // passing over, a blurred grab — is a gap in the evidence, not
        // evidence against it. A board that actually changed reads different
        // letters, and that resets the streak on its own.
        drawOverlay();
        setStatus(deep ? "Looking for the board…" : "Point the camera at the board");
        renderDiagnostics();
        recordFrame(phases, deep, iterationStart, consensus.progress());
        await nextFrame();
        continue;
      }

      quickLookMisses = 0;
      drawOverlay();

      const classifyStart = performance.now();
      let predictions: Prediction[];
      try {
        const classifier = await getClassifier();
        predictions = await classifier.classifyCells(result.cells);
      } catch (error) {
        // Unlike readFrame's catch, this used to be unguarded: a failure here
        // (classifier.classifyCells retries once internally and can still
        // throw, e.g. two backends both unavailable) would escape the loop
        // entirely — the camera stayed live with nothing reading it, which
        // looks exactly like a freeze with no error ever shown.
        console.error("scanner classify failed", error);
        diagnostics.note = `classify failed: ${String(error).slice(0, 90)}`;
        renderDiagnostics();
        await nextFrame();
        continue;
      }
      phases.classifyMs = performance.now() - classifyStart;
      diagnostics.classifyMs = phases.classifyMs;
      if (token !== loopToken) return;

      const locked = consensus.add({
        letters: predictions.map((p) => (p.label === "?" ? "" : p.label)),
        confidences: predictions.map((p) => p.confidence),
      });

      const progress = consensus.progress();
      diagnostics.frames = progress.frames;
      diagnostics.unsettled = progress.unsettled;
      diagnostics.meanConfidence = progress.meanConfidence;
      diagnostics.note = "";
      renderDiagnostics();
      recordFrame(phases, deep, iterationStart, progress);

      if (locked) {
        lockIn(locked, result.fit.quad, frame, result.warped, result.cells, result.full, result.quad);
        return; // camera is stopped inside lockIn; nothing left for this loop to do
      } else {
        setStatus(
          progress.unsettled > 0
            ? `Reading… ${progress.unsettled} of 25 still unsettled`
            : `Reading… hold steady (confidence ${progress.meanConfidence.toFixed(2)})`,
        );
      }
      await nextFrame();
    }
  }

  /** Yields to the browser between frames so the preview keeps painting.
   * Detection and classification are async but GPU-bound, and without a real
   * yield the compositor can starve. */
  function nextFrame(): Promise<void> {
    // requestAnimationFrame does not fire in a backgrounded tab, which would
    // park the loop forever mid-scan while still holding the camera. The
    // timeout is the escape hatch; the loop's own visibility check stops it.
    return new Promise((resolve) => {
      let settled = false;
      // Cancel the loser. Leaving both pending left a stray timer (and a
      // retained closure) behind on every single frame of the scan; the loop
      // runs for as long as the camera is up, so "small per frame" is the
      // only kind of leak this file can have.
      const done = () => {
        if (settled) return;
        settled = true;
        cancelAnimationFrame(rafHandle);
        clearTimeout(timerHandle);
        resolve();
      };
      const rafHandle = requestAnimationFrame(done);
      const timerHandle = setTimeout(done, 250);
    });
  }

  function stop(): void {
    wantsToRun = false;
    running = false;
    startInitiatedAt = null;
    loopToken++;
    if (pendingStop !== null) {
      clearTimeout(pendingStop);
      pendingStop = null;
    }
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
    video.srcObject = null;
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  async function start(): Promise<void> {
    wantsToRun = true;
    if (pendingStop !== null) {
      clearTimeout(pendingStop);
      pendingStop = null;
    }
    if (running) return;
    if (starting) return starting;
    startInitiatedAt = performance.now();
    starting = startInner().finally(() => {
      starting = null;
    });
    return starting;
  }

  async function startInner(): Promise<void> {
    // Kick the downloads off before asking for the camera: the permission
    // prompt and stream negotiation are dead time the fetches can hide behind.
    prefetchModels();
    consensus.reset();
    quickLookMisses = 0;
    setStatus("Starting camera…");

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          // Ask for detail: a die glyph is a small part of the frame, and the
          // classifier's crops come straight out of it.
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
    } catch (error) {
      // Show why, on screen. "Nothing happened" is not a diagnosable report,
      // and the reasons are very different: NotAllowedError is permission (or
      // an untrusted certificate), NotFoundError is no camera, NotReadableError
      // is another app holding it.
      const name = error instanceof Error ? error.name : "Error";
      const message = error instanceof Error ? error.message : String(error);
      console.warn("camera unavailable", error);
      setStatus("Camera unavailable.");
      showError(
        name === "NotAllowedError"
          ? `Camera blocked (${name}). On a self-signed certificate some browsers refuse the camera even after you accept the warning.`
          : `Camera error: ${name} — ${message}`,
        true,
      );
      diagnostics.note = `getUserMedia: ${name}`;
      renderDiagnostics();
      startInitiatedAt = null;
      return;
    }

    clearError();
    // Defensive: never overwrite a live stream reference without stopping it.
    video.srcObject = stream;
    // Autoplay can reject; without this the loop would spin on a 0x0 video
    // forever with no indication of why.
    try {
      await video.play();
    } catch (error) {
      const name = error instanceof Error ? error.name : "Error";
      showError(`Preview did not start (${name}). Tap Retry.`, true);
      diagnostics.note = `video.play: ${name}`;
      renderDiagnostics();
    }

    running = true;
    startInitiatedAt = null;
    autoRecoveries = 0;
    lastProgressAt = performance.now();
    writeHeartbeat({ phase: "running" });
    startupMs = performance.now();
    // performance.now() is already measured from page load, so the scanner's
    // start time *is* the startup cost.
    telemetry.begin();
    void loop(++loopToken);
  }

  /** Forces the camera loop back to a known state after the watchdog decides
   * nothing has progressed in too long — see WATCHDOG_STALL_MS. Abandons any
   * stuck `starting`/`loop` in place rather than waiting on it (a hung
   * promise can't be cancelled), tears down whatever camera state exists,
   * and tries once more, bounded by MAX_AUTO_RECOVERIES so a device that
   * genuinely can't hold a stream doesn't retry forever in the background. */
  function recoverFromStall(reason: string): void {
    console.warn(`scanner watchdog: ${reason}`);
    autoRecoveries++;
    writeHeartbeat({ phase: "recovered-from-stall", reason, autoRecoveries });
    starting = null;
    startInitiatedAt = null;
    stop();
    if (autoRecoveries > MAX_AUTO_RECOVERIES) {
      showError("The camera keeps stalling on this device. Reload the page.", true);
      diagnostics.note = `watchdog: gave up after ${autoRecoveries} recoveries`;
      renderDiagnostics();
      return;
    }
    diagnostics.note = `watchdog: recovered (${reason})`;
    renderDiagnostics();
    wantsToRun = true;
    void start();
  }

  setInterval(() => {
    const now = performance.now();
    if (starting && startInitiatedAt !== null && now - startInitiatedAt > WATCHDOG_STALL_MS) {
      recoverFromStall("start() never resolved (likely a stuck getUserMedia)");
    } else if (running && now - lastProgressAt > WATCHDOG_STALL_MS) {
      recoverFromStall("camera loop stopped making progress");
    } else if (running) {
      writeHeartbeat({ phase: "running" });
    }
  }, WATCHDOG_CHECK_MS);

  // Release the camera when the page is hidden or torn down. Without this the
  // track stayed live past a reload: the browser holds it until the old
  // context is collected, and the next getUserMedia can block or fail with
  // NotReadableError while the camera light stays on.
  //
  // pagehide (not beforeunload/unload) is the one that fires reliably on
  // mobile, including bfcache navigations.
  window.addEventListener("pagehide", () => {
    stop();
    // The one case that counts as a clean exit for the stale-heartbeat check
    // — every other path that reaches here (backgrounding, a watchdog
    // recovery) deliberately leaves a breadcrumb behind instead.
    clearHeartbeat();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden" && running) {
      // A backgrounded tab cannot paint, so a scan there is pure battery
      // drain on a stream the user cannot see. Pause rather than stall —
      // `wantsToRun` survives this so returning to the tab resumes on its own.
      sendTelemetry("cancelled");
      const shouldResume = wantsToRun;
      stop();
      wantsToRun = shouldResume;
      // Distinct from the "running"/"recovered-from-stall" phases: an OS
      // reclaiming a backgrounded tab is expected, not evidence of the freeze
      // this is meant to catch. Left in place (not cleared) so a resume that
      // never gets going again still leaves *some* trace of the last known
      // state, just not an alarming one.
      writeHeartbeat({ phase: "backgrounded" });
    } else if (document.visibilityState === "visible" && wantsToRun && !running) {
      void start();
    }
  });

  retryBtn.addEventListener("click", () => {
    clearError();
    void start();
  });

  reviewBtn.addEventListener("click", () => {
    boardView.toggleReview();
    reviewBtn.textContent = boardView.reviewing ? "Show board" : "Review letters";
  });

  /** Sends the current lock to `/api/submit` as a flagged-wrong training
   * sample: the full-res photo plus the board *as read*, since there is no
   * manual-correction UI to supply a fix (see the project brief on the UI
   * strip-down). `photo.toBlob` reads the canvas's pixels synchronously at
   * this call, before any `await`, so a "Scan a new board" tap racing the
   * upload can't corrupt it — `frameCanvas` gets reused by the next lock. */
  reportBtn.addEventListener("click", () => {
    const report = pendingReport;
    if (!report) return;
    reportBtn.disabled = true;
    reportBtn.textContent = "Reporting…";
    report.photo.toBlob(
      (blob) => {
        if (!blob) {
          reportBtn.disabled = false;
          reportBtn.textContent = "Couldn't report — tap to retry";
          return;
        }
        const meta = {
          letters: report.letters,
          predictions: report.letters,
          confidences: report.confidences,
          quad: report.quad,
          frameWidth: report.frameWidth,
          frameHeight: report.frameHeight,
          gridSize: state.gridSize,
          reportedWrong: true,
        };
        const form = new FormData();
        form.append("photo", blob, "photo.jpg");
        form.append("meta", JSON.stringify(meta));
        fetch(SUBMIT_ENDPOINT, { method: "POST", body: form })
          .then((response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            reportBtn.textContent = "Reported — thanks";
          })
          .catch((error) => {
            console.warn("submit report failed", error);
            reportBtn.disabled = false;
            reportBtn.textContent = "Couldn't reach server — tap to retry";
          });
      },
      "image/jpeg",
      0.92,
    );
  });

  rescanBtn.addEventListener("click", () => {
    rescanBtn.hidden = true;
    reviewBtn.hidden = true;
    reportBtn.hidden = true;
    pendingReport = null;
    boardView.reset();
    video.style.opacity = "1";
    void start();
  });

  return {
    start,
    stop,
    showPath: (path) => boardView.showPath(path),
    showHeatmap: (weights) => boardView.showHeatmap(weights),
    playFlourish: (paths, totalMs) => boardView.playFlourish(paths, totalMs),
  };
}
