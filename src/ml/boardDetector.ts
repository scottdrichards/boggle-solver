import * as tf from "@tensorflow/tfjs";
import { detectGrid, type GridDetection, type GridDetectorOptions } from "../cv/gridDetector";
import type { PixelBuffer } from "../cv/quadWarp";
import { ensureBackend } from "./backend";

const INPUT_SIZE = 128;

/**
 * Loads the cell-centre heatmap model and runs the board detector against it.
 *
 * Kept separate from `LetterClassifier` because the two models are used at
 * different moments — detection the instant a frame arrives, classification
 * only once the board is located — and each should cost its own download no
 * earlier than needed. They share one backend, chosen in `backend.ts`.
 */
export class BoardDetector {
  private model: tf.LayersModel | null = null;

  async load(modelUrl: string): Promise<void> {
    await ensureBackend();
    this.model = await tf.loadLayersModel(modelUrl);
    // Warm up: the detector's first call is on the user's critical path, and
    // shader compilation would otherwise land there. Warmed at a batch of 1
    // and again at a realistic batch, since some backends compile per shape.
    for (const count of [1, 8]) {
      const warmup = tf.zeros([count, INPUT_SIZE, INPUT_SIZE, 1]);
      const result = this.model.predict(warmup) as tf.Tensor;
      await result.data();
      tf.dispose([warmup, result]);
    }
  }

  async detect(source: PixelBuffer, options: GridDetectorOptions = {}): Promise<GridDetection | null> {
    const model = this.model;
    if (!model) throw new Error("board detector not loaded");

    return detectGrid(
      source,
      async (batch, count) => {
        const output = tf.tidy(
          () => model.predict(tf.tensor4d(batch, [count, INPUT_SIZE, INPUT_SIZE, 1])) as tf.Tensor,
        );
        // One readback for the whole batch. This is the expensive part of a
        // forward pass at this model size — it stalls until the GPU drains —
        // so the batching upstream exists precisely to do it once per level
        // rather than once per crop.
        const heatmaps = Float32Array.from(await output.data());
        output.dispose();
        return heatmaps;
      },
      options,
    );
  }
}
