/**
 * The backend name list and the page-scoped override reader, split out of
 * `backend.ts` specifically to have **zero import of `@tensorflow/tfjs`**.
 *
 * Why this earns its own file: `pipelineClient.ts` (main thread) needs
 * `requestedBackend()` to resolve the page's `?backend=`/`localStorage`
 * override before handing it to the pipeline worker (see that file's doc
 * comment) — but it must never import anything that drags tf.js into the
 * main thread's bundle, which is the entire point of moving the CV/ML
 * pipeline into a worker in the first place. Importing `requestedBackend`
 * straight from `backend.ts` used to do exactly that: Vite's chunking
 * shares a module across every entry point that reaches it, so pulling in
 * one small function from `backend.ts` (which does `import * as tf from
 * "@tensorflow/tfjs"` at module scope) pulled tf.js's ~730 KB core chunk
 * into `index.html`'s `modulepreload` list — verified by inspecting the
 * built `dist/` output, not assumed. `backend.ts` re-exports everything
 * here so its own callers (`classifier.ts`, `boardDetector.ts`,
 * `pipeline.worker.ts`, `benchmark.ts`) don't need to change.
 */

export const BACKEND_CHAIN = ["webgpu", "webgl", "wasm", "cpu"] as const;
export type BackendName = (typeof BACKEND_CHAIN)[number];

/** Explicit override, for benchmarking and for pinning a device that turns
 * out to have a bad driver. Unrecognised values are ignored.
 *
 * Reads `location`/`localStorage`, both page-scoped — meaningless if called
 * from inside a worker (a worker's own `location` is its script URL, not the
 * page's, and `localStorage` doesn't exist there at all; the latter already
 * degrades safely via the existing try/catch, but the former would silently
 * never see a page's `?backend=` param). `pipeline.worker.ts` never calls
 * this directly for that reason — see `backend.ts`'s `ensureBackend`'s
 * `forced` param. */
export function requestedBackend(): BackendName | null {
  const fromQuery = new URLSearchParams(location.search).get("backend");
  const stored = (() => {
    try {
      return localStorage.getItem("boggle.backend");
    } catch {
      return null; // Private mode / storage disabled — not worth failing over.
    }
  })();
  const name = fromQuery ?? stored;
  return BACKEND_CHAIN.includes(name as BackendName) ? (name as BackendName) : null;
}
