/**
 * On-device backend benchmark. Open `benchmark.html` on the phone you
 * actually scan with.
 *
 * This exists because "use WebGPU, it's the GPU" is an assumption, not a
 * measurement, and at this model size it is genuinely in doubt: the detector
 * is 128x128 in / 32x32 out and the classifier is 158k parameters over 32x32
 * tiles, so per-call dispatch and the blocking readback of the result are a
 * large fraction of the cost. WASM+SIMD pays neither, and can win outright.
 * Rather than argue, run all four and read the table.
 *
 * What it measures, per backend:
 *  - detector forward pass at batch 1, 8 and 24 — the ratio between these is
 *    the whole case for batching the pyramid. Perfectly parallel hardware
 *    would show batch 24 costing the same as batch 1.
 *  - a full `detectGrid` on a real photo, which is what a scanner frame costs.
 *  - the classifier on 25 cells with 4-rotation TTA, i.e. one whole board.
 */
import * as tf from "@tensorflow/tfjs";
import { detectGrid } from "./cv/gridDetector";
import type { PixelBuffer } from "./cv/quadWarp";
import { BACKEND_CHAIN, resetBackendSelection, type BackendName } from "./ml/backend";

const DETECTOR_INPUT = 128;
const CLASSIFIER_INPUT = 32;
const WORKING_MAX_DIMENSION = 1400;
/** Repeats per timing. The first run of anything is shader compilation, so
 * every measurement below discards a warm-up pass and reports the median. */
const REPEATS = 5;

interface Row {
  backend: BackendName;
  label: string;
  ms: number | null;
  note?: string;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/** Times `run` REPEATS times after one discarded warm-up, returning the median.
 * `run` must await its own readback — timing a dispatch you never read back
 * measures queueing, not work. */
async function time(run: () => Promise<unknown>): Promise<number> {
  await run();
  const samples: number[] = [];
  for (let i = 0; i < REPEATS; i++) {
    const start = performance.now();
    await run();
    samples.push(performance.now() - start);
  }
  return median(samples);
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

/** The demo photo, downscaled exactly as the app's working canvas does. */
async function loadWorkingBuffer(url: string): Promise<PixelBuffer> {
  const image = await loadImage(url);
  const scale = Math.min(1, WORKING_MAX_DIMENSION / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.naturalWidth * scale);
  canvas.height = Math.round(image.naturalHeight * scale);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function activate(backend: BackendName): Promise<void> {
  if (backend === "webgpu") {
    await import("@tensorflow/tfjs-backend-webgpu");
  } else if (backend === "wasm") {
    const wasm = await import("@tensorflow/tfjs-backend-wasm");
    wasm.setWasmPaths(`${import.meta.env.BASE_URL}tfjs-wasm/`);
  }
  await tf.setBackend(backend);
  await tf.ready();
}

async function benchmarkBackend(backend: BackendName, photo: PixelBuffer, report: (row: Row) => void): Promise<void> {
  const base = import.meta.env.BASE_URL;
  const detector = await tf.loadLayersModel(`${base}cell-heatmap/model.json`);
  const classifier = await tf.loadLayersModel(`${base}model/model.json`);

  for (const count of [1, 8, 24]) {
    const input = tf.zeros([count, DETECTOR_INPUT, DETECTOR_INPUT, 1]);
    const ms = await time(async () => {
      const output = detector.predict(input) as tf.Tensor;
      await output.data();
      output.dispose();
    });
    input.dispose();
    report({ backend, label: `detector forward, batch ${count}`, ms, note: `${(ms / count).toFixed(2)} ms/crop` });
  }

  // The real thing: a whole frame through the pyramid. Reported with the
  // dispatch count, since that is what the batching change moves.
  let batches = 0;
  let passes = 0;
  const detectMs = await time(async () => {
    const fit = await detectGrid(
      photo,
      async (batch, count) => {
        const output = tf.tidy(
          () => detector.predict(tf.tensor4d(batch, [count, DETECTOR_INPUT, DETECTOR_INPUT, 1])) as tf.Tensor,
        );
        const data = Float32Array.from(await output.data());
        output.dispose();
        return data;
      },
      { gridSizes: [5] },
    );
    batches = fit?.batches ?? 0;
    passes = fit?.passes ?? 0;
  });
  report({ backend, label: "detectGrid, real photo", ms: detectMs, note: `${batches} dispatches / ${passes} crops` });

  // 25 cells x 4 rotations, the classifier's actual per-board workload.
  const cells = tf.zeros([25, CLASSIFIER_INPUT, CLASSIFIER_INPUT, 1]);
  const classifyMs = await time(async () => {
    const probabilities = tf.tidy(() => {
      let view = cells as tf.Tensor4D;
      let total: tf.Tensor | null = null;
      for (let k = 0; k < 4; k++) {
        const probs = classifier.predict(view) as tf.Tensor;
        total = total === null ? probs : tf.add(total, probs);
        view = tf.reverse(tf.transpose(view, [0, 2, 1, 3]), [1]) as tf.Tensor4D;
      }
      return tf.div(total!, 4);
    });
    await probabilities.data();
    probabilities.dispose();
  });
  cells.dispose();
  report({ backend, label: "classifier, 25 cells x4 TTA", ms: classifyMs });

  detector.dispose();
  classifier.dispose();
}

async function main(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#benchmark")!;
  root.innerHTML = `
    <header class="app-header">
      <h1>Backend benchmark</h1>
      <p class="dictionary-status" id="status">Loading photo…</p>
    </header>
    <table class="benchmark-table">
      <thead><tr><th>backend</th><th>measurement</th><th>median ms</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  `;
  const status = root.querySelector<HTMLElement>("#status")!;
  const rows = root.querySelector<HTMLElement>("#rows")!;

  const report = (row: Row): void => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.backend}</td>
      <td>${row.label}</td>
      <td class="numeric">${row.ms === null ? "—" : row.ms.toFixed(1)}</td>
      <td class="note">${row.note ?? ""}</td>
    `;
    rows.append(tr);
  };

  const photo = await loadWorkingBuffer(`${import.meta.env.BASE_URL}demo/board-working.jpg`);

  for (const backend of BACKEND_CHAIN) {
    status.textContent = `Running ${backend}…`;
    try {
      await activate(backend);
      await benchmarkBackend(backend, photo, report);
    } catch (error) {
      // An unavailable backend is a result, not a failure: "no WebGPU on this
      // phone" is exactly what this page is for finding out.
      report({ backend, label: "unavailable", ms: null, note: String(error).slice(0, 120) });
    }
  }

  // Leaving a benchmark run's backend selected would silently pin the app to
  // whatever ran last, so hand the choice back.
  resetBackendSelection();
  status.textContent = "Done. Lower is better; compare within a column, not across devices.";
  status.classList.add("ready");
}

void main();
