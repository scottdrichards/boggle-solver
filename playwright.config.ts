import { defineConfig, devices } from "@playwright/test";

// Real headless Chromium (Playwright's bundled build), not a system browser.
// GPU is software-rendered (SwiftShader) since this box has no display GPU —
// WebGL works, WebGPU does not, matching the app's own webgpu->webgl->wasm
// fallback chain. See ../CLAUDE.md "Not yet verified in a real browser".
//
// Software rendering is also much slower than any real device this project
// has measured (see CLAUDE.md's on-device benchmark table) — a full
// coarse-to-fine pyramid pass here can run past the scanner's own 15s
// watchdog stall threshold, which then force-cycles the camera and the scan
// never converges. So the fixture used for a "does it actually detect a
// real board" test is deliberately cropped tight around the board (see
// board-clean.y4m below) so the cheap quick-look-only path finds it, rather
// than relying on the deep pyramid fallback surviving under this box's
// unusually slow rendering.

/** Chrome's fake video capture device takes exactly one file per browser
 * launch (`--use-file-for-fake-video-capture`), so a spec that needs a real
 * board in view gets its own project pointed at its own fixture, scoped to
 * just that spec via `testMatch` — projects can't share a launch flag, and a
 * spec run under the wrong project would "detect" whatever the wrong video
 * shows instead of the board it was written against. */
function fakeCameraProject(name: string, fixture: string, spec: string) {
  return {
    name,
    testMatch: spec,
    use: {
      ...devices["Desktop Chrome"],
      // Localhost counts as a secure context, so getUserMedia works over
      // plain http here without the TLS server the CLAUDE.md notes require
      // for on-device testing.
      permissions: ["camera"],
      launchOptions: {
        args: [
          "--use-fake-device-for-media-stream",
          "--use-fake-ui-for-media-stream",
          "--use-file-for-fake-video-capture=" + new URL(`./e2e/fixtures/${fixture}`, import.meta.url).pathname,
        ],
      },
    },
  };
}

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  retries: 0,
  // First model load + shader compile under this box's software-rendered
  // WebGL (no real GPU) measured at ~13-14s — give real headroom. The live
  // board-scan spec sets its own longer per-test timeout on top of this.
  timeout: 45_000,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5183",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npx vite --port 5183 --strictPort",
    url: "http://localhost:5183",
    reuseExistingServer: true,
    timeout: 60_000,
  },
  projects: [
    // Generic SMPTE color-bars test pattern — no board in frame. Only checks
    // that the app boots, the camera opens and the dictionary loads.
    fakeCameraProject("chromium", "fake-camera.y4m", "scanner.spec.ts"),
    // A real board photo (public/demo/board-working.jpg), cropped tight
    // around the tray so it fills most of the frame — see the file header
    // above for why. Exercises the whole pipeline end-to-end in a real
    // browser for the first time: detect, lock, classify, solve.
    fakeCameraProject("chromium-scan-board", "board-clean.y4m", "scan-board.spec.ts"),
  ],
});
