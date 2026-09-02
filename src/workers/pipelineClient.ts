import type { PixelBuffer } from "../cv/quadWarp";
// Deliberately from `backendChain.ts`, not `backend.ts` — see that file's
// doc comment. Importing from `backend.ts` here would pull `@tensorflow/tfjs`
// into the main thread's bundle via Vite's shared-chunk resolution, exactly
// what moving the pipeline into a worker is meant to avoid.
import { requestedBackend, type BackendName } from "../ml/backendChain";
import type {
  FitSummary,
  FrameMessage,
  PipelineFramePhases,
  InitMessage,
  PipelineResponse,
  PredictionSummary,
} from "./pipelineProtocol";

/** Main-thread handle to the CV/ML pipeline worker (`pipeline.worker.ts`).
 * Mirrors `solverClient.ts`'s request/response bookkeeping, plus the part
 * that file doesn't need: a worker that owns a camera-rate loop's GPU state
 * has to be killable and restartable, not just message-able. See
 * `terminate()`/`ensureStarted()`.
 *
 * Deliberately never imports `ml/backend.ts`'s `ensureBackend`/tfjs, or
 * `ml/models.ts`/`ml/classifier.ts` — none of that runs on the main thread
 * any more. This class only imports `requestedBackend()` (page-scoped, reads
 * `location`/`localStorage`, must run here) and the tfjs-free types.
 */
export interface PipelineFrameResult {
  fit: FitSummary | null;
  /** Detect-resolution frame dimensions — always present, even on a miss,
   * since `BoardLockParams` (`boardView.ts`) needs the same coordinate space
   * `fit.quad` is expressed in. */
  detectWidth: number;
  detectHeight: number;
  warped: PixelBuffer | null;
  cells: PixelBuffer[] | null;
  predictions: PredictionSummary[] | null;
  phases: Partial<PipelineFramePhases>;
}

interface PendingRequest {
  resolve(result: PipelineFrameResult): void;
  reject(error: Error): void;
}

export class PipelineClient {
  private worker: Worker | null = null;
  private nextRequestId = 0;
  private readonly pending = new Map<number, PendingRequest>();
  private _backend: BackendName | null = null;
  private _tensors = { numTensors: 0, numBytes: 0 };
  private _fullResolution: { width: number; height: number } | null = null;

  /** `null` until the worker's first reply — scanner.ts shows "loading" for
   * this gap the same way it always did while `activeBackend()` was still
   * `null` pre-migration. */
  get backend(): BackendName | null {
    return this._backend;
  }

  get tensorStats(): { numTensors: number; numBytes: number } {
    return this._tensors;
  }

  /** Working-resolution dimensions from the most recent found board — `null`
   * until a board has been found at least once, since that draw only happens
   * then. Feeds `ScanReport.workingResolution` (`telemetry.ts`). */
  get fullResolution(): { width: number; height: number } | null {
    return this._fullResolution;
  }

  get isLive(): boolean {
    return this.worker !== null;
  }

  /** Creates the worker if it isn't already live and kicks off model
   * loading — fire-and-forget, same call-site contract as the old
   * `prefetchModels()`: call it right before `getUserMedia` so the
   * permission prompt and stream negotiation hide the download/shader-
   * compile latency. Safe to call repeatedly; a no-op once live.
   *
   * Deliberately swallows a synchronous construction failure rather than
   * letting it propagate — this is called from `scanner.ts`'s `startInner()`
   * without a surrounding try/catch (it's meant to be fire-and-forget, like
   * the `prefetchModels()` call it replaced), and an uncaught throw there
   * would silently kill the whole `start()` promise chain before
   * `getUserMedia` ever runs, with no error shown and nothing left to
   * trigger the watchdog's recovery path (both `starting` and `running`
   * settle to falsy with no error surfaced) — a worse failure mode than
   * anything that existed before this worker did. `this.worker` stays
   * `null` on failure, so the next `runFrame`/`ensureStarted()` call just
   * retries. */
  ensureStarted(): void {
    if (this.worker) return;
    try {
      const worker = new Worker(new URL("./pipeline.worker.ts", import.meta.url), { type: "module" });
      worker.addEventListener("message", (event: MessageEvent<PipelineResponse>) => {
        this.handleMessage(event.data);
      });
      worker.addEventListener("error", (event: ErrorEvent) => {
        console.error("pipeline worker error", event.message);
        // An error here means whatever the worker was doing when it happened
        // is not going to finish — anyone awaiting a frame from it would hang
        // forever otherwise. Treat it as a termination: fail pending, drop the
        // reference, and let the caller's own retry path (scanner.ts's
        // watchdog, or the next `runFrame`) recreate it.
        this.terminate();
      });
      this.worker = worker;
      // Absolute, not relative: `import.meta.env.BASE_URL` ("./" in this app)
      // only resolves correctly against a known document location, and the
      // worker's own script URL is the wrong one — see InitMessage's doc
      // comment and ml/models.ts's `assetBase` for the bug this avoids.
      const assetBase = new URL(import.meta.env.BASE_URL, location.href).href;
      const init: InitMessage = { type: "init", backendOverride: requestedBackend(), assetBase };
      worker.postMessage(init);
    } catch (error) {
      console.error("pipeline worker failed to start", error);
      this.worker = null;
    }
  }

  /** Forcibly kills the worker. This is the actual point of moving the
   * pipeline off the main thread: an abandoned main-thread `await` can never
   * be cancelled, only ignored, but `Worker.terminate()` really does stop
   * whatever the worker was doing, deterministically, even mid-GPU-call. Any
   * frame request still in flight is rejected — scanner.ts's `loop()` treats
   * that exactly like the old code treated a `readFrame`/`classifyCells`
   * throw: the frame is skipped, nothing is recorded for it. Safe to call
   * when already stopped. */
  terminate(): void {
    if (!this.worker) return;
    this.worker.terminate();
    this.worker = null;
    this._backend = null;
    for (const request of this.pending.values()) request.reject(new Error("pipeline worker terminated"));
    this.pending.clear();
  }

  private handleMessage(message: PipelineResponse): void {
    switch (message.type) {
      case "ready":
        this._backend = message.backend;
        return;
      case "load-error":
        console.error("pipeline worker failed to load models", message.message);
        return;
      case "miss": {
        this._backend = message.backend;
        this._tensors = { numTensors: message.tensors, numBytes: message.tensorMB * 1e6 };
        const request = this.pending.get(message.requestId);
        if (!request) return;
        this.pending.delete(message.requestId);
        request.resolve({
          fit: null,
          detectWidth: message.detectWidth,
          detectHeight: message.detectHeight,
          warped: null,
          cells: null,
          predictions: null,
          phases: message.phases,
        });
        return;
      }
      case "result": {
        this._backend = message.backend;
        this._tensors = { numTensors: message.tensors, numBytes: message.tensorMB * 1e6 };
        this._fullResolution = { width: message.fullWidth, height: message.fullHeight };
        const request = this.pending.get(message.requestId);
        if (!request) return;
        this.pending.delete(message.requestId);
        request.resolve({
          fit: message.fit,
          detectWidth: message.detectWidth,
          detectHeight: message.detectHeight,
          warped: message.warped,
          cells: [...message.cells],
          predictions: [...message.predictions],
          phases: message.phases,
        });
        return;
      }
      case "frame-error": {
        const request = this.pending.get(message.requestId);
        if (!request) return;
        this.pending.delete(message.requestId);
        request.reject(new Error(`pipeline ${message.phase} failed: ${message.message}`));
      }
    }
  }

  /** Runs one frame through detection (and warp+slice+classify, if a board
   * was found) in the worker. Captures the frame itself via
   * `createImageBitmap(video)` — deliberately not a canvas draw + readback:
   * that CPU-bound pair is exactly what used to cost the main thread on
   * every single frame, and creating a bitmap doesn't pay it (the worker
   * does the readback instead, off this thread). Returns `null` if the video
   * has nothing to give yet (0x0), matching the old `captureInto`'s "not
   * ready" signal so the caller can skip the iteration without it counting
   * as a miss. */
  async runFrame(video: HTMLVideoElement, deep: boolean): Promise<PipelineFrameResult | null> {
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;
    this.ensureStarted();

    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(video);
    } catch {
      return null; // e.g. the track was stopped mid-capture — treat as "not ready"
    }

    const worker = this.worker;
    if (!worker) {
      bitmap.close();
      return null; // terminated between ensureStarted() and the bitmap resolving
    }

    const requestId = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      const message: FrameMessage = { type: "frame", requestId, bitmap, deep };
      worker.postMessage(message, [bitmap]);
    });
  }
}
