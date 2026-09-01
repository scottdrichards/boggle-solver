import { expect, test } from "@playwright/test";

// First real-browser verification of this app (see ../CLAUDE.md "Not yet
// verified in a real browser" / "no browser on the agent box" — Playwright's
// bundled Chromium resolves that: getUserMedia over localhost is a secure
// context, and --use-fake-device-for-media-stream feeds it e2e/fixtures/
// fake-camera.y4m in place of a physical camera.

test("app boots, camera opens, dictionary loads", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(String(err)));

  await page.goto("/?debug=1");

  // Dictionary status only becomes visible on failure — absence of the
  // error text is the pass condition, not a positive "ready" element.
  await expect(page.locator("#dictionary-status")).toBeHidden();

  const video = page.locator("#scanner-video");
  await expect(video).toBeVisible();
  await expect
    .poll(async () => video.evaluate((el: HTMLVideoElement) => el.readyState))
    .toBeGreaterThanOrEqual(2); // HAVE_CURRENT_DATA: the fake device is actually streaming

  const errorPanel = page.locator("#scanner-error");
  await expect(errorPanel).toBeHidden();

  // Diagnostics (?debug=1) prove the pipeline is actually running each
  // frame, not just that the <video> tag exists. First model load + shader
  // compile under this box's software-rendered WebGL (no real GPU here)
  // measured at ~13-14s, well beyond a real device — give it real headroom.
  const diagnostics = page.locator("#scanner-diagnostics");
  await expect(diagnostics).toBeVisible();
  await expect
    .poll(() => diagnostics.textContent(), { timeout: 30_000 })
    .toMatch(/backend/i);

  expect(consoleErrors, consoleErrors.join("\n")).toEqual([]);
});
