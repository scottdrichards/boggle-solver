import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        // Two pages: the app, and the on-device backend benchmark.
        main: fileURLToPath(new URL("index.html", import.meta.url)),
        benchmark: fileURLToPath(new URL("benchmark.html", import.meta.url)),
      },
    },
  },
  test: {
    environment: "node",
  },
});
