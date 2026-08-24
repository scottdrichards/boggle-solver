import * as tf from "@tensorflow/tfjs";
import { assessCellQuality } from "../cv/cellQuality";
import { preprocessCellForModel } from "../cv/preprocess";
import type { PixelBuffer } from "../cv/quadWarp";
import { activeBackend, ensureBackend, fallbackBackend, type BackendName } from "./backend";

export interface Prediction {
  label: string;
  confidence: number;
}

/** Sentinel label for "no confident letter" — same convention scanner.ts already
 * maps to an abstaining vote, whether the cause is a low softmax or a rejected crop. */
const NO_READING: Prediction = { label: "?", confidence: 0 };

const INPUT_SIZE = 32;

/**
 * Rotates a batch of [N,H,W,1] tiles 90 degrees counter-clockwise. Exact for
 * square tiles — pure index permutation, no resampling.
 */
function rot90(x: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => tf.reverse(tf.transpose(x, [0, 2, 1, 3]), [1]) as tf.Tensor4D);
}

/**
 * Averages the classifier over the four right-angle rotations of each cell.
 *
 * A die's printed letter is the same whichever way the die was dropped, so all
 * four rotations are equally valid views of the same label — averaging them is
 * variance reduction, not relabelling.
 *
 * This matters more than it sounds. Dice are not uniformly oriented in
 * practice (they tend to sit roughly aligned with the tray), so the classifier
 * picks up an orientation prior from the training photos — and that prior is
 * wrong by 90 degrees the moment the phone is held in the other aspect.
 * Measured on 56 held-out boards (1400 cells) shot in landscape where the
 * training photos were portrait: 92.0% upright-only vs 99.5% with this
 * averaging. The previous model showed the same effect (86.3% -> 94.4%), so it
 * is a property of the pipeline, not of one set of weights.
 *
 * Costs 4x inference on a 158k-parameter model over 25 cells, which is
 * negligible next to grid detection.
 */
function predictRotationAveraged(model: tf.LayersModel, batch: tf.Tensor4D): tf.Tensor2D {
  return tf.tidy(() => {
    let view = batch;
    let total: tf.Tensor2D | null = null;
    for (let k = 0; k < 4; k++) {
      const probs = model.predict(view) as tf.Tensor2D;
      total = total === null ? probs : (tf.add(total, probs) as tf.Tensor2D);
      view = rot90(view);
    }
    return tf.div(total!, 4) as tf.Tensor2D;
  });
}

/**
 * Loads the trained letter classifier and runs batched inference. The backend
 * is chosen once for the whole app in `backend.ts` — for a batch this small
 * (<=25 tiny 32x32 images) GPU dispatch/transfer overhead can plausibly cost
 * more than it saves, so that choice is "best available", not an assumption
 * that GPU wins.
 */
export class LetterClassifier {
  private model: tf.LayersModel | null = null;
  private labels: string[] = [];
  private modelUrl = "";
  private backend: BackendName = "cpu";

  async load(modelUrl: string, labelsUrl: string): Promise<void> {
    this.modelUrl = modelUrl;
    const labelsResponse = await fetch(labelsUrl);
    if (!labelsResponse.ok) throw new Error(`failed to load labels from ${labelsUrl}: ${labelsResponse.status}`);
    this.labels = (await labelsResponse.json()) as string[];

    this.backend = await ensureBackend();
    await this.loadModelOnCurrentBackend();
  }

  get activeBackend(): string {
    return activeBackend() ?? this.backend;
  }

  private async loadModelOnCurrentBackend(): Promise<void> {
    // Release the previous weights first. This method is called again on a
    // mid-session backend fallback, and without this the old model's
    // variables stayed live — on WebGL/WebGPU that is retained GPU memory on
    // a device that just proved it was under pressure.
    this.model?.dispose();
    this.model = null;
    this.model = await tf.loadLayersModel(this.modelUrl);
    // Warm-up inference hides WebGPU/WebGL's first-use shader-compile
    // latency so the user's first real classification isn't the slow one.
    const warmup = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 1]);
    const result = this.model.predict(warmup) as tf.Tensor;
    await result.data();
    tf.dispose([warmup, result]);
  }

  private async runBatch(cells: readonly PixelBuffer[]): Promise<Prediction[]> {
    const model = this.model;
    if (!model) throw new Error("classifier not loaded");

    const batchData = new Float32Array(cells.length * INPUT_SIZE * INPUT_SIZE);
    cells.forEach((cell, i) => {
      batchData.set(preprocessCellForModel(cell, INPUT_SIZE), i * INPUT_SIZE * INPUT_SIZE);
    });

    const batch = tf.tensor4d(batchData, [cells.length, INPUT_SIZE, INPUT_SIZE, 1]);
    let predictions: tf.Tensor2D;
    try {
      predictions = predictRotationAveraged(model, batch);
    } finally {
      batch.dispose();
    }

    const probsByRow = (await predictions.array()) as number[][];
    predictions.dispose();

    return probsByRow.map((row, i) => {
      // A glare-blown or washed-out crop is rejected before we even ask the
      // model — the softmax on a corrupted glyph is frequently confident, so
      // confidence alone can't be trusted to catch this (that's the whole
      // reason the frame-level confidence gate in consensus.ts isn't enough).
      if (assessCellQuality(cells[i]!).rejected) return NO_READING;

      let bestIndex = 0;
      for (let j = 1; j < row.length; j++) if (row[j]! > row[bestIndex]!) bestIndex = j;
      return { label: this.labels[bestIndex] ?? "?", confidence: row[bestIndex]! };
    });
  }

  /** Classifies a batch of cropped board-cell images in one pass. */
  async classifyCells(cells: readonly PixelBuffer[]): Promise<Prediction[]> {
    if (!this.model) throw new Error("classifier not loaded");
    try {
      return await this.runBatch(cells);
    } catch (error) {
      // Covers e.g. a WebGPU "device lost" mid-session: move to the next
      // backend and reload (tensors from a stale backend don't carry over).
      console.warn("classification failed, switching backend and retrying", error);
      this.backend = await fallbackBackend(this.backend);
      await this.loadModelOnCurrentBackend();
      return await this.runBatch(cells);
    }
  }
}
