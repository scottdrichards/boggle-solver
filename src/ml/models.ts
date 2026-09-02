/**
 * Lazy singletons for the two models.
 *
 * Lazy because tfjs and the weights shouldn't cost anything on initial page
 * load — only once a scan actually starts. Singletons because the scanner and
 * the photo path both want them, and loading a second copy would mean a
 * second download and a second set of compiled kernels on the GPU.
 */
import type { BoardDetector } from "./boardDetector";
import type { LetterClassifier } from "./classifier";

let detectorPromise: Promise<BoardDetector> | null = null;
let classifierPromise: Promise<LetterClassifier> | null = null;

/** Where model files are fetched from. Defaults to `import.meta.env.BASE_URL`
 * (a *relative* path — this app builds with `base: "./"`), which is correct
 * as long as whatever imports this module runs on the main thread: a
 * relative fetch there resolves against the page's own URL.
 *
 * `pipeline.worker.ts` overrides this via `setAssetBase()` before ever
 * calling `getDetector`/`getClassifier`, because that assumption breaks
 * inside a worker — a worker's relative fetches resolve against *its own
 * script's* URL, which in a production build lives one directory deeper
 * (under `assets/`) than the page. `./cell-heatmap/model.json` fetched from
 * inside the worker landed on the wrong path, fell through to the SPA's HTML
 * fallback, and failed to parse as JSON — a real bug that shipped once (see
 * this app's CLAUDE.md, "the worker's own location isn't the page's" — this
 * is the same class of mistake, just in a different file's fetch calls) and
 * only shows up in a production build, never in `vite dev`, since dev-server
 * routing happens not to introduce the extra directory level. */
let assetBase = import.meta.env.BASE_URL;

/** Sets an absolute base URL for model fetches — see `assetBase`'s doc
 * comment. Must be called before the first `getDetector`/`getClassifier`
 * call to have any effect (both memoize their promise on first call). */
export function setAssetBase(base: string): void {
  assetBase = base;
}

export function getDetector(): Promise<BoardDetector> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const { BoardDetector } = await import("./boardDetector");
      const detector = new BoardDetector();
      await detector.load(`${assetBase}cell-heatmap/model.json`);
      return detector;
    })();
  }
  return detectorPromise;
}

export function getClassifier(): Promise<LetterClassifier> {
  if (!classifierPromise) {
    classifierPromise = (async () => {
      const { LetterClassifier } = await import("./classifier");
      const classifier = new LetterClassifier();
      await classifier.load(`${assetBase}model/model.json`, `${assetBase}model/labels.json`);
      return classifier;
    })();
  }
  return classifierPromise;
}

/** Starts both downloads without waiting. The scanner calls this the moment
 * it opens: the camera permission prompt and the stream negotiation are dead
 * time that the model fetches can hide behind. */
export function prefetchModels(): void {
  void getDetector().catch(() => {});
  void getClassifier().catch(() => {});
}
