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

export function getDetector(): Promise<BoardDetector> {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const { BoardDetector } = await import("./boardDetector");
      const detector = new BoardDetector();
      await detector.load(`${import.meta.env.BASE_URL}cell-heatmap/model.json`);
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
      const base = import.meta.env.BASE_URL;
      await classifier.load(`${base}model/model.json`, `${base}model/labels.json`);
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
