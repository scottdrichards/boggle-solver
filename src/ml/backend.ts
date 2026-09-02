/**
 * One TFJS backend, shared by both models.
 *
 * This used to be private to `LetterClassifier`, which meant the *detector*
 * — the heavier of the two models, and the one on the critical path the
 * instant a photo appears — never opted in at all: `BoardDetector.load()`
 * called `tf.ready()` without ever calling `setBackend`, so it silently took
 * whatever tfjs defaulted to (WebGL) while the WebGPU package was imported
 * only from the classifier's side. Whichever model loads first now decides
 * for both, and they decide the same way.
 *
 * Order is "best available", not a claim that GPU always wins. For batches
 * this small the dispatch and readback overhead is a real fraction of the
 * cost, and WASM+SIMD has none of it — see `benchmark.ts`, which exists to
 * settle this per-device rather than by argument. Override with
 * `?backend=wasm` (or localStorage `boggle.backend`) to force one.
 */
import * as tf from "@tensorflow/tfjs";
import { BACKEND_CHAIN, requestedBackend, type BackendName } from "./backendChain";

// Re-exported so existing callers (classifier.ts, boardDetector.ts,
// pipeline.worker.ts, benchmark.ts) don't need to change imports — only
// `pipelineClient.ts` (main thread) imports `backendChain.ts` directly,
// specifically to avoid pulling tf.js in with it. See that file's doc
// comment for why the split exists.
export { BACKEND_CHAIN, requestedBackend, type BackendName };

/** Where the WASM binaries are served from. They are copied out of
 * node_modules into `public/tfjs-wasm/` by `scripts/copyWasm.ts` — tfjs
 * fetches them at runtime by name, so they cannot be bundled.
 *
 * Defaults to `import.meta.env.BASE_URL` (relative — this app builds with
 * `base: "./"`), correct only when resolved on the main thread. Inside a
 * worker a relative path resolves against the worker script's own URL, not
 * the page's — see `ml/models.ts`'s `assetBase` doc comment for the full
 * story (same bug, same fix, this is the WASM-path half of it).
 * `pipeline.worker.ts` calls `setAssetBase()` before touching this. */
let assetBase = import.meta.env.BASE_URL;

export function setAssetBase(base: string): void {
  assetBase = base;
}

let loadedWasm = false;

async function activate(backend: BackendName): Promise<void> {
  if (backend === "webgpu") {
    await import("@tensorflow/tfjs-backend-webgpu");
  } else if (backend === "wasm") {
    const wasm = await import("@tensorflow/tfjs-backend-wasm");
    if (!loadedWasm) {
      // Threaded WASM additionally needs cross-origin isolation (COOP/COEP),
      // which GitHub Pages cannot send, so in production this resolves to the
      // single-threaded SIMD build. tfjs picks that up on its own from the
      // failed threads probe; we only have to serve all three binaries.
      wasm.setWasmPaths(`${assetBase}tfjs-wasm/`);
      loadedWasm = true;
    }
  }
  await tf.setBackend(backend);
  await tf.ready();
}

/** A `requestDevice()`/`setBackend()` call has no timeout of its own, and on
 * some Android drivers a GPU context request made right after the tab's GPU
 * process was reclaimed while backgrounded can apparently hang rather than
 * reject (unconfirmed — no browser on the agent box to reproduce with, see
 * scanner.ts's watchdog for the same caveat). A stuck backend must not be
 * allowed to block the whole chain forever; treat a timeout as this
 * candidate failing so the loop moves on to the next one. */
const BACKEND_ACTIVATE_TIMEOUT_MS = 8000;

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function selectFrom(startIndex: number): Promise<BackendName> {
  let lastError: unknown;
  for (let i = startIndex; i < BACKEND_CHAIN.length; i++) {
    const backend = BACKEND_CHAIN[i]!;
    try {
      await withTimeout(activate(backend), BACKEND_ACTIVATE_TIMEOUT_MS, `activate(${backend})`);
      return backend;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`no working TFJS backend available: ${String(lastError)}`);
}

let selection: Promise<BackendName> | null = null;

/** Resolves once a backend is live. Memoized: both models call it, and the
 * second caller must not re-run backend init while the first is mid-flight.
 *
 * `forced`, when passed, skips `requestedBackend()` entirely and uses this
 * value instead — how `pipeline.worker.ts` honours the page's `?backend=`/
 * `localStorage` override despite not being able to read either itself: the
 * main thread resolves `requestedBackend()` and sends the result in over
 * `postMessage`. Omit it (main-thread callers, benchmark.ts, tests) to keep
 * reading the page directly, as before. */
export function ensureBackend(forced?: BackendName | null): Promise<BackendName> {
  if (!selection) {
    selection = (async () => {
      const requested = forced !== undefined ? forced : requestedBackend();
      if (requested) {
        try {
          // Same timeout as the automatic chain below — this path had none
          // until now, a real gap: a forced backend (via `?backend=` or a
          // stale `localStorage` value from earlier testing) that hangs
          // instead of failing cleanly would block `ensureBackend()` forever
          // with nothing to fall back to and no way for a caller to notice.
          await withTimeout(activate(requested), BACKEND_ACTIVATE_TIMEOUT_MS, `activate(${requested})`);
          return requested;
        } catch (error) {
          console.warn(`forced backend "${requested}" unavailable, falling back`, error);
        }
      }
      return selectFrom(0);
    })();
  }
  return selection;
}

/** Moves to the next backend down the chain after a mid-session failure
 * (e.g. a WebGPU "device lost"). Callers must reload their models: tensors
 * and compiled kernels do not carry across backends. */
export function fallbackBackend(after: BackendName): Promise<BackendName> {
  selection = selectFrom(BACKEND_CHAIN.indexOf(after) + 1);
  return selection;
}

/** The backend currently in use, or null before `ensureBackend` resolves. */
export function activeBackend(): BackendName | null {
  const name = tf.getBackend();
  return BACKEND_CHAIN.includes(name as BackendName) ? (name as BackendName) : null;
}

/** Live tensor count and bytes held by the backend.
 *
 * Exposed because a tensor leak in a loop that runs for as long as the camera
 * is up is invisible from anywhere else: tfjs holds GPU textures/buffers that
 * neither the JS heap profile nor `performance.memory` accounts for, and a
 * phone has no console to check `tf.memory()` from. A count that climbs frame
 * over frame is a missing `dispose`/`tidy`; one that sits flat rules the whole
 * class of cause out in a single glance. */
export function tensorMemory(): { numTensors: number; numBytes: number } {
  const memory = tf.memory();
  return { numTensors: memory.numTensors, numBytes: memory.numBytes };
}

/** Test/benchmark hook: forget the memoized choice so the next
 * `ensureBackend()` re-selects. Not used by the app itself. */
export function resetBackendSelection(): void {
  selection = null;
}
