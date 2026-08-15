/**
 * Copies the TFJS WASM binaries into `public/tfjs-wasm/`.
 *
 * tfjs fetches these by filename at runtime (see `setWasmPaths` in
 * `src/ml/backend.ts`), so they cannot go through the bundler. All three
 * variants ship: tfjs probes for threads and SIMD support and picks one, and
 * which it picks depends on the browser and on whether the page is
 * cross-origin isolated.
 */
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist");
const destination = join(here, "..", "public", "tfjs-wasm");

const files = [
  "tfjs-backend-wasm.wasm",
  "tfjs-backend-wasm-simd.wasm",
  "tfjs-backend-wasm-threaded-simd.wasm",
];

mkdirSync(destination, { recursive: true });
for (const file of files) {
  copyFileSync(join(source, file), join(destination, file));
}
console.log(`copied ${files.length} wasm binaries to public/tfjs-wasm/`);
