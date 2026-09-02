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
 *
 * 4. **Detection and classification run in a worker, not here** (2026-09-01
 *    — see `../workers/pipeline.worker.ts`). This file's only per-frame
 *    pixel work is capturing an `ImageBitmap` and handing it off; every
 *    tfjs/CV cost that used to block this thread's rendering mid-scan now
 *    happens off it, and a stuck GPU call can be killed outright
 *    (`pipeline.terminate()`) instead of just abandoned. `pipeline` is a
 *    `PipelineClient`, and its lifecycle is deliberately different from the
 *    camera's: `stop()` releases the camera immediately on backgrounding,
 *    but the worker gets a longer grace period before it's actually killed
 *    (`PIPELINE_IDLE_MS`) — see the `visibilitychange` handler below.
 */
import type { PixelBuffer, Point } from "../cv/quadWarp";
import { PipelineClient, type PipelineFrameResult } from "../workers/pipelineClient";
import type { PipelineFramePhases } from "../workers/pipelineProtocol";
import { ScanConsensus, type LockedReading } from "./consensus";
import { ScanTelemetry } from "./telemetry";
import { mountBoardView, LOCK_TRANSITION_MS } from "./boardView";
import { setGridSize, setTiles, state, type GridSize } from "./state";

/** `cellSubmitServer.py`'s endpoints (see that script + the boggle brief's
 * "abuse-resistant cell submission" entry) — a small stdlib server proxied
 * at `/boggle/api/*` on the projects.scottdrichards.com box. Relative +
 * BASE_URL-prefixed so this resolves correctly wherever the app is served;
 * on GitHub Pages (static, no backend) the POST just fails, which the report
 * handler treats the same as any other network error. */
const TOKEN_ENDPOINT = `${import.meta.env.BASE_URL}api/token`;
const SUBMIT_CELLS_ENDPOINT = `${import.meta.env.BASE_URL}api/submit-cells`;
/** Flagged crops are a training sample, not a photo — small and cheap is
 * strictly better here: the model's own input is 32x32, so 96px at modest
 * JPEG quality already carries far more detail than the classifier uses,
 * while keeping a full board's worth of flags a trivial upload even on a
 * bad connection. */
const CELL_EXPORT_SIZE = 96;
const CELL_EXPORT_QUALITY = 0.7;
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

/** How long a backgrounded tab keeps the CV/ML pipeline worker warm before
 * it's actually killed. This is a separate, longer-fused decision from the
 * camera itself: `stop()` releases the camera hardware immediately on
 * backgrounding (pure battery drain otherwise, see the `visibilitychange`
 * handler below), but the worker holds loaded tfjs models and compiled GPU
 * kernels, which are comparatively expensive to reacquire — a `?debug=1`
 * device measured 13-14s for first load under this project's slowest tested
 * backend. Killing it on every brief backgrounding (a notification pull-down,
 * a quick app switch) would trade a rare memory saving for a routine reload
 * stall. Killing it *never* would defeat the point of moving the pipeline to
 * a worker in the first place — an extended background stay should actually
 * free the GPU/wasm memory, not just pause. This is the balance: reuse a
 * still-warm worker across a quick switch, but let a real stay behind
 * another app or a locked screen actually spin it down. */
const PIPELINE_IDLE_MS = 20000;

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

export function mountScanner(container: HTMLElement, handlers: ScannerHandlers): ScannerController {
  container.innerHTML = `
    <div class="scanner-stage" id="scanner-stage">
      <video id="scanner-video" playsinline muted autoplay></video>
      <canvas id="scanner-overlay"></canvas>
      <div class="scanner-status" id="scanner-status">Starting camera…</div>
    </div>
    <p class="scanner-error" id="scanner-error" hidden></p>
    <div class="scanner-actions">
      <button type="button" id="scanner-switch-camera" hidden>Switch camera</button>
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
  const switchCameraBtn = container.querySelector<HTMLButtonElement>("#scanner-switch-camera")!;
  const retryBtn = container.querySelector<HTMLButtonElement>("#scanner-retry")!;
  const reviewBtn = container.querySelector<HTMLButtonElement>("#scanner-review")!;
  const reportBtn = container.querySelector<HTMLButtonElement>("#scanner-report")!;
  const rescanBtn = container.querySelector<HTMLButtonElement>("#scanner-rescan")!;
  const diagnosticsEl = container.querySelector<HTMLElement>("#scanner-diagnostics")!;
  const overlayCtx = overlay.getContext("2d")!;
  // Presentation after a lock: freezing the frame, animating the quad into a
  // flat square, and drawing a tapped word's trail on it. See boardView.ts.
  // `onFlagsChanged` is how the review grid's tap-to-flag cells (also in
  // boardView.ts) reach the report button below.
  const boardView = mountBoardView(stage, video, {
    onFlagsChanged: (indices) => updateReportButton(indices),
  });

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

  // The live video is 3:4 portrait, but the board it captures is always
  // square (boggle/board.ts). Once locked there's no video left to frame —
  // just the flattened square board — so the stage itself switches to a 1:1
  // aspect instead of staying 3:4 with the square letterboxed inside it.
  // Without this the locked board sat centered in a 3:4 box with black bars
  // above and below, and the status text over those bars was barely legible.
  let stageAspect = STAGE_ASPECT;

  function computeStageSize(targetHeightVh: number): { width: number; height: number } {
    const columnWidth = container.clientWidth || stage.clientWidth || 1;
    const heightCapPx = (targetHeightVh / 100) * window.innerHeight;
    const height = Math.min(columnWidth / stageAspect, heightCapPx);
    return { width: height * stageAspect, height };
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

  // Detection, warping, slicing and classification all happen in the CV/ML
  // pipeline worker now — see pipelineClient.ts/pipeline.worker.ts and this
  // file's header comment. The main thread's only per-frame pixel work is
  // handing the worker an `ImageBitmap`; there is no capture canvas here to
  // draw on by accident any more, which makes the old "never draw on the
  // capture surface" invariant structural rather than just documented.
  const pipeline = new PipelineClient();
  /** Scheduled by the `visibilitychange` handler when the tab is hidden —
   * see PIPELINE_IDLE_MS. Cleared if the tab becomes visible again first. */
  let pipelineIdleTimer: number | null = null;

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
  /** Every `videoinput` device seen since the last `getUserMedia` call.
   * Labels (and a full list at all, on some browsers) only populate once
   * permission has been granted at least once, so this starts empty and is
   * refreshed inside `startInner()` after the first successful stream. */
  let videoDevices: MediaDeviceInfo[] = [];
  let currentDeviceIndex = -1;
  /** Set by `switchCamera()`; `null` means "use the default facing-camera
   * constraint", same as before this feature existed. */
  let desiredDeviceId: string | null = null;
  let switchingCamera = false;
  let loopToken = 0;
  let quickLookMisses = 0;
  /** Board size the current run of consensus votes assumes. The fitter picks
   * a size fresh every frame (see `pipeline.worker.ts`'s `GRID_SIZES`), so a
   * flip mid-scan is possible — treated the same as the board itself
   * changing: the vote window can't mix readings of two different sizes, so
   * it resets rather than silently mis-indexing cells. */
  let scanGridSize: number | null = null;
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

  /** Snapshot of the current lock's per-cell crops, kept only so tapping a
   * cell wrong in the review grid (boardView.ts) has something to submit —
   * there is no manual-correction UI, so a report says "this crop is NOT
   * this letter" rather than supplying the right one. Only the *flagged*
   * cells are ever uploaded (see reportBtn below): a full-board photo is
   * neither necessary for that signal nor as cheap to store. */
  let pendingCells: {
    cells: readonly PixelBuffer[];
    letters: string[];
    confidences: (number | null)[];
    gridSize: number;
  } | null = null;

  /** Short-lived anti-abuse token from `cellSubmitServer.py`'s `/api/token`
   * — proves a submission came from a client that actually loaded the app
   * recently, not a bare script hitting the upload endpoint blind. Fetched
   * once per scanner start and again lazily if a submit finds it missing or
   * the server rejects it as expired; a failed fetch just leaves this null,
   * and the submit attempt below surfaces that as a normal network error
   * rather than crashing anything. */
  let submitToken: string | null = null;

  async function fetchSubmitToken(): Promise<void> {
    try {
      const response = await fetch(TOKEN_ENDPOINT);
      if (!response.ok) return;
      const body = (await response.json()) as { token?: string };
      submitToken = body.token ?? null;
    } catch {
      // No backend behind this origin (e.g. GitHub Pages) — reporting will
      // fail with the same "couldn't reach server" message it always had.
      submitToken = null;
    }
  }

  /** Downscales one classifier-input crop to a small JPEG for upload. Cheap
   * relative to a full board photo by design — see `CELL_EXPORT_SIZE`. */
  function exportCell(cell: PixelBuffer): Promise<Blob | null> {
    const canvas = document.createElement("canvas");
    canvas.width = cell.width;
    canvas.height = cell.height;
    canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(cell.data), cell.width, cell.height), 0, 0);
    const small = document.createElement("canvas");
    small.width = CELL_EXPORT_SIZE;
    small.height = CELL_EXPORT_SIZE;
    const smallCtx = small.getContext("2d")!;
    smallCtx.imageSmoothingEnabled = true;
    smallCtx.imageSmoothingQuality = "high";
    smallCtx.drawImage(canvas, 0, 0, CELL_EXPORT_SIZE, CELL_EXPORT_SIZE);
    return new Promise((resolve) => small.toBlob(resolve, "image/jpeg", CELL_EXPORT_QUALITY));
  }

  /** Reflects the review grid's current flags onto the report button — shown
   * only once at least one cell is flagged, since "report nothing" isn't an
   * action. Re-derived from `boardView.getFlaggedCells()` on every tap
   * rather than tracked separately here. */
  function updateReportButton(indices: readonly number[]): void {
    if (indices.length === 0) {
      reportBtn.hidden = true;
      return;
    }
    reportBtn.hidden = false;
    reportBtn.disabled = false;
    reportBtn.textContent = `Report ${indices.length} wrong letter${indices.length === 1 ? "" : "s"}`;
  }

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
    gridCells: 25,
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
      }${diagnostics.deep ? ", deep" : ""})  inliers ${diagnostics.inliers}/${diagnostics.gridCells}`,
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
    phases: Partial<PipelineFramePhases>,
    deep: boolean,
    iterationStart: number,
    progress: { unsettled: number; meanConfidence: number },
  ): void {
    // Tensor accounting now lives in the pipeline worker (that's where tfjs
    // runs) — the client just mirrors whatever the worker last reported,
    // same as `activeBackend()`/`tensorMemory()` used to read module-level
    // state directly on this thread pre-migration.
    const tensors = pipeline.tensorStats;
    // Non-standard and Chromium-only; absent everywhere else, hence the cast
    // rather than a global type declaration for something that may not exist.
    // Still read on the main thread deliberately: this is *this* thread's
    // heap, not the worker's, and a scan can die from either — see
    // telemetry.ts's `heapMB` doc comment.
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
    const full = pipeline.fullResolution;
    telemetry.send(
      {
        userAgent: navigator.userAgent,
        backend: pipeline.backend ?? "unknown",
        videoResolution: `${video.videoWidth}x${video.videoHeight}`,
        workingResolution: full ? `${full.width}x${full.height}` : "0x0",
        startupMs: startupMs,
        outcome,
      },
      letters,
    );
  }

  function setStatus(text: string): void {
    statusEl.textContent = text;
    statusEl.hidden = false;
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

  /** Pauses the video (so its last frame stays visible under the flatten
   * animation) and stops the read loop immediately, but only releases the
   * camera hardware once the animation has had time to fully cover it —
   * releasing sooner would blank the video mid-transition, since clearing
   * `srcObject` drops the last frame instantly in most browsers. */
  function lockIn(
    locked: LockedReading,
    quadInFrame: readonly [Point, Point, Point, Point],
    detectWidth: number,
    detectHeight: number,
    warped: PixelBuffer,
    cells: readonly PixelBuffer[],
  ): void {
    setTiles(locked.letters);
    sendTelemetry("locked", locked.letters);

    // The exact crops the classifier saw for this lock — reporting only ever
    // sends whichever of these the player taps wrong in the review grid, so
    // this is kept, not the full-resolution photo the old whole-board report
    // used to be attached to.
    pendingCells = {
      cells,
      letters: [...locked.letters],
      confidences: [...locked.confidences],
      gridSize: state.gridSize,
    };
    reportBtn.hidden = true; // shown once a cell is actually flagged wrong
    switchCameraBtn.hidden = true; // nothing to switch once the camera stops

    wantsToRun = false;
    running = false;
    loopToken++;
    video.pause();

    boardView.lock({
      quad: quadInFrame,
      frameWidth: detectWidth,
      frameHeight: detectHeight,
      board: warped,
      gridSize: state.gridSize,
      cells,
      letters: locked.letters,
      confidences: locked.confidences,
    });
    reviewBtn.hidden = false;
    reviewBtn.textContent = "Review letters";
    rescanBtn.hidden = false;
    // No status text once locked — the flattened board and the results list
    // say everything that mattered, and the text used to hang over the
    // stage's now-removed black bars, barely legible against them.
    statusEl.hidden = true;
    stageAspect = 1;
    applyStageSize();

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

      const deep = quickLookMisses >= QUICK_LOOKS_BEFORE_PYRAMID;
      diagnostics.backend = pipeline.backend ?? "loading";
      diagnostics.resolution = `${video.videoWidth}x${video.videoHeight}`;
      diagnostics.deep = deep;

      let result: PipelineFrameResult | null;
      try {
        result = await pipeline.runFrame(video, deep);
      } catch (error) {
        // Covers both a detect-phase and a classify-phase failure in the
        // worker (see pipelineClient.ts's `frame-error` handling) — same
        // treatment either way the original code gave each separately: log
        // it (it used to vanish into a console a phone user could never see),
        // skip the frame, and let the loop keep going.
        if (token !== loopToken) return;
        console.error("scanner frame failed", error);
        diagnostics.note = `pipeline failed: ${String(error).slice(0, 90)}`;
        renderDiagnostics();
        await nextFrame();
        continue;
      }
      if (token !== loopToken) return;

      if (!result) {
        // Video not ready yet (0x0) — not a miss, just nothing to read.
        await nextFrame();
        continue;
      }

      diagnostics.detectMs = result.phases.detectMs ?? 0;
      diagnostics.inliers = result.phases.inliers ?? 0;
      diagnostics.dispatches = result.phases.dispatches ?? 0;

      if (!result.fit || !result.cells || !result.warped || !result.predictions) {
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
        recordFrame(result.phases, deep, iterationStart, consensus.progress());
        await nextFrame();
        continue;
      }

      quickLookMisses = 0;
      diagnostics.gridCells = result.fit.gridSize * result.fit.gridSize;
      if (result.fit.gridSize !== scanGridSize) {
        // The window can't mix votes for two different sizes — a "cell 5" on
        // a 4-wide board and a 5-wide board isn't the same square. Treated
        // like the board itself changing size mid-scan (rare, but the fit is
        // a fresh guess every frame): drop the old votes and start settling
        // the newly-sized board instead.
        consensus.reset();
        scanGridSize = result.fit.gridSize;
      }
      setGridSize(result.fit.gridSize as GridSize);
      drawOverlay();
      diagnostics.classifyMs = result.phases.classifyMs ?? 0;

      const locked = consensus.add({
        letters: result.predictions.map((p) => (p.label === "?" ? "" : p.label)),
        confidences: result.predictions.map((p) => p.confidence),
      });

      const progress = consensus.progress();
      diagnostics.frames = progress.frames;
      diagnostics.unsettled = progress.unsettled;
      diagnostics.meanConfidence = progress.meanConfidence;
      diagnostics.note = "";
      renderDiagnostics();
      recordFrame(result.phases, deep, iterationStart, progress);

      if (locked) {
        lockIn(locked, result.fit.quad, result.detectWidth, result.detectHeight, result.warped, result.cells);
        return; // camera is stopped inside lockIn; nothing left for this loop to do
      } else {
        setStatus(
          progress.unsettled > 0
            ? `Reading… ${progress.unsettled} of ${progress.cells.length} still unsettled`
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

  /** Requests the camera, preferring whatever `desiredDeviceId` was set to by
   * `switchCamera()` — the same default-facing-camera constraint as before
   * this feature existed when it's `null`. If the chosen device fails (e.g.
   * unplugged, or permission for it revoked mid-session), falls back to the
   * default once rather than leaving the scanner stuck on a dead device until
   * a reload. */
  function acquireStream(): Promise<MediaStream> {
    const base = {
      // Ask for detail: a die glyph is a small part of the frame, and the
      // classifier's crops come straight out of it.
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    };
    if (!desiredDeviceId) {
      return navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, ...base },
        audio: false,
      });
    }
    return navigator.mediaDevices
      .getUserMedia({ video: { deviceId: { exact: desiredDeviceId }, ...base }, audio: false })
      .catch(() => {
        desiredDeviceId = null;
        return navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, ...base },
          audio: false,
        });
      });
  }

  /** Refreshes the known camera list and shows/hides the switch button.
   * `enumerateDevices()` only returns real labels (and, on some browsers, the
   * full list at all) once permission has been granted at least once, so this
   * is called after every successful `getUserMedia`, not just at mount. */
  async function refreshDeviceList(activeStream: MediaStream): Promise<void> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      videoDevices = devices.filter((d) => d.kind === "videoinput");
    } catch {
      videoDevices = [];
    }
    const activeId = activeStream.getVideoTracks()[0]?.getSettings().deviceId ?? null;
    currentDeviceIndex = activeId ? videoDevices.findIndex((d) => d.deviceId === activeId) : -1;
    switchCameraBtn.hidden = videoDevices.length < 2;
  }

  /** Cycles to the next known camera (front/back, or a phone's extra rear
   * lenses) and restarts the stream on it. Only the camera track is torn down
   * and reacquired — the CV/ML pipeline worker stays warm, and `startInner`'s
   * own `consensus.reset()` clears votes for the new viewpoint same as any
   * other restart. */
  async function switchCamera(): Promise<void> {
    if (switchingCamera || videoDevices.length < 2) return;
    switchingCamera = true;
    switchCameraBtn.disabled = true;
    currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
    desiredDeviceId = videoDevices[currentDeviceIndex]!.deviceId;
    stop();
    try {
      await start();
    } finally {
      switchingCamera = false;
      switchCameraBtn.disabled = false;
    }
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
    // Also cancels any pending idle-termination of the pipeline worker (see
    // the `visibilitychange` handler below) — a no-op if it's already live.
    if (pipelineIdleTimer !== null) {
      clearTimeout(pipelineIdleTimer);
      pipelineIdleTimer = null;
    }
    pipeline.ensureStarted();
    if (!submitToken) void fetchSubmitToken();
    consensus.reset();
    quickLookMisses = 0;
    setStatus("Starting camera…");

    try {
      stream = await acquireStream();
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
      switchCameraBtn.hidden = true;
      startInitiatedAt = null;
      return;
    }

    clearError();
    void refreshDeviceList(stream);
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
   * camera-side promise still can't be cancelled), tears down whatever
   * camera state exists, and tries once more, bounded by MAX_AUTO_RECOVERIES
   * so a device that genuinely can't hold a stream doesn't retry forever in
   * the background.
   *
   * The pipeline worker gets different treatment: `pipeline.terminate()`
   * actually cancels whatever it was doing, deterministically, instead of
   * being abandoned. This is the reliable half of moving the CV/ML pipeline
   * off the main thread — before, a stuck detect/classify call inside
   * `readFrame` could only be left running forever; now the stall recovery
   * can kill it outright and `startInner()`'s `pipeline.ensureStarted()`
   * spins up a clean one on the retry below. */
  function recoverFromStall(reason: string): void {
    console.warn(`scanner watchdog: ${reason}`);
    autoRecoveries++;
    writeHeartbeat({ phase: "recovered-from-stall", reason, autoRecoveries });
    starting = null;
    startInitiatedAt = null;
    stop();
    pipeline.terminate();
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
    // No grace period here, unlike the visibilitychange path below — the
    // page is actually going away, not just hidden, so there is no "quick
    // switch back" case to optimise for.
    if (pipelineIdleTimer !== null) {
      clearTimeout(pipelineIdleTimer);
      pipelineIdleTimer = null;
    }
    pipeline.terminate();
    // The one case that counts as a clean exit for the stale-heartbeat check
    // — every other path that reaches here (backgrounding, a watchdog
    // recovery) deliberately leaves a breadcrumb behind instead.
    clearHeartbeat();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      if (running) {
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
      }
      // The pipeline worker gets a longer, separate fuse than the camera —
      // see PIPELINE_IDLE_MS. Scheduled whenever the worker is actually live,
      // regardless of whether a scan was running: a worker left warm after a
      // lock (waiting for "Scan a new board") holds exactly the same loaded-
      // model memory while backgrounded as one mid-scan.
      if (pipeline.isLive && pipelineIdleTimer === null) {
        pipelineIdleTimer = window.setTimeout(() => {
          pipelineIdleTimer = null;
          pipeline.terminate();
        }, PIPELINE_IDLE_MS);
      }
    } else if (document.visibilityState === "visible") {
      // Back before the grace period elapsed — the worker (if still live)
      // needs no reload at all; if it already got terminated, `start()`'s
      // `pipeline.ensureStarted()` recreates it lazily.
      if (pipelineIdleTimer !== null) {
        clearTimeout(pipelineIdleTimer);
        pipelineIdleTimer = null;
      }
      if (wantsToRun && !running) void start();
    }
  });

  retryBtn.addEventListener("click", () => {
    clearError();
    void start();
  });

  switchCameraBtn.addEventListener("click", () => {
    void switchCamera();
  });

  reviewBtn.addEventListener("click", () => {
    boardView.toggleReview();
    reviewBtn.textContent = boardView.reviewing ? "Show board" : "Review letters";
  });

  /** Sends only the cells the player tapped wrong in the review grid to
   * `cellSubmitServer.py`'s `/api/submit-cells` — small JPEGs, not a full
   * board photo, and gated by `submitToken` so a bare script hitting the
   * endpoint without ever loading the app gets rejected server-side. There
   * is no manual-correction UI, so this reports "not this letter" rather
   * than supplying the right one — see `pendingCells`'s comment. */
  reportBtn.addEventListener("click", async () => {
    const pending = pendingCells;
    const indices = boardView.getFlaggedCells();
    if (!pending || indices.length === 0) return;
    reportBtn.disabled = true;
    reportBtn.textContent = "Reporting…";

    if (!submitToken) await fetchSubmitToken();
    if (!submitToken) {
      reportBtn.disabled = false;
      reportBtn.textContent = "Couldn't reach server — tap to retry";
      return;
    }

    try {
      const form = new FormData();
      const meta = indices.map((index) => ({
        index,
        letter: pending.letters[index] ?? "",
        confidence: pending.confidences[index] ?? null,
      }));
      form.append("meta", JSON.stringify({ gridSize: pending.gridSize, cells: meta }));
      for (const index of indices) {
        const blob = await exportCell(pending.cells[index]!);
        if (blob) form.append(`cell_${index}`, blob, `cell_${index}.jpg`);
      }
      const response = await fetch(SUBMIT_CELLS_ENDPOINT, {
        method: "POST",
        headers: { "X-Boggle-Token": submitToken },
        body: form,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      reportBtn.textContent = "Reported — thanks";
    } catch (error) {
      console.warn("submit report failed", error);
      reportBtn.disabled = false;
      reportBtn.textContent = "Couldn't reach server — tap to retry";
    }
  });

  rescanBtn.addEventListener("click", () => {
    rescanBtn.hidden = true;
    reviewBtn.hidden = true;
    reportBtn.hidden = true;
    pendingCells = null;
    boardView.reset();
    video.style.opacity = "1";
    stageAspect = STAGE_ASPECT;
    applyStageSize();
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
