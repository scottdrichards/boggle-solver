# Boggle Solver

A browser app that photographs a physical Boggle board, reads the 25 letters
with an on-device vision pipeline, and finds every valid word — no server,
no upload, everything runs locally in the browser (TensorFlow.js on
WebGPU/WebGL/WASM).

Point your camera at a 5x5 Boggle board. The app detects the grid, classifies
each die face, and once several consecutive frames agree, locks in the board
and shows the solution.

## How it works

1. **Detection** — a small CNN predicts a cell-centre heatmap over the frame;
   the peaks are fit to a 5x5 lattice (homography-based, RANSAC-style search)
   to find the board's quad and per-cell centres even at an angle.
2. **Classification** — each cell is warped to a square, sliced out, and
   classified by a second CNN trained on real and rendered die-face crops.
   Predictions are averaged over all 4 right-angle rotations (a die's letter
   is orientation-invariant, but the training data isn't), which is a large,
   cheap accuracy win.
3. **Consensus** — each of the 25 cells votes independently over a rolling
   window of recent frames. The board locks once every cell has a stable
   majority reading above a confidence floor — voting per-cell (not
   per-board) means one flickery die can't block the other 24 from settling.
4. **Solving** — a word-search DFS over a trie built from an English word
   list, respecting Boggle's adjacency and single-use-per-tile rules.

## Running locally

```sh
npm install
npm run dev
```

`getUserMedia` (camera access) requires a secure context, so testing the
scanner from a phone needs HTTPS — `npm run dev` over plain HTTP works fine
on desktop, or serve a build behind any TLS-terminating host for on-device
testing.

## Building

```sh
npm run build
```

Builds two pages: the app (`index.html`) and `benchmark.html`, a diagnostics
page that times the ML backends (WebGPU/WebGL/WASM/CPU) at several batch
sizes on whatever device opens it — useful for judging which backend a given
phone should prefer.

## Testing

```sh
npm run test
npm run typecheck
```

## Tech

TypeScript, Vite, TensorFlow.js. No backend, no build-time secrets — the
GitHub Actions workflow in `.github/workflows/deploy.yml` builds and deploys
straight to GitHub Pages on push to `main`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
