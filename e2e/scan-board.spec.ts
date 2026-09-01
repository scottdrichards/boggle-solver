import { expect, test } from "@playwright/test";

// First real-browser confirmation that the scanner actually detects and
// locks a real board, not just that the camera opens (see scanner.spec.ts).
// Runs under the "chromium-scan-board" project (playwright.config.ts), whose
// fake camera streams e2e/fixtures/board-clean.y4m — a tight crop of the
// tray from public/demo/board-working.jpg, cropped specifically so the board
// fills most of the frame (see playwright.config.ts's header comment for
// why: this box's software-rendered WebGL is far slower than any real
// device this project has measured, and a full photo can push detection
// into the deep-pyramid fallback, which then loses a race against the
// scanner's own 15s stall watchdog under that slowdown).
//
// The true letters on this board were never transcribed anywhere in the
// repo, so this can't assert an exact word list — it asserts the pipeline
// actually ran end-to-end: a lock happened, the classifier produced 25
// confident letters, and the solver found real words on them.

test("detects, locks and solves a real board photo", async ({ page }) => {
  test.setTimeout(60_000);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(String(err)));

  await page.goto("/?debug=1");

  // A lock is the "Scan a new board" button appearing — see lockIn() in
  // scanner.ts. Model load/shader compile plus several quick-look frames
  // under software-rendered WebGL: give it real headroom.
  const rescanBtn = page.locator("#scanner-rescan");
  await expect(rescanBtn).toBeVisible({ timeout: 50_000 });

  // The status text is deliberately hidden once locked (see this app's
  // CLAUDE.md, "unsightly black bars" fix) — its absence is itself part of
  // what a clean lock looks like, not just something to ignore.
  await expect(page.locator("#scanner-status")).toBeHidden();

  const diagnostics = page.locator("#scanner-diagnostics");
  await expect(diagnostics).toContainText(/inliers 2[0-5]\/25/); // near-full 5x5 board

  const summary = page.locator(".results-summary");
  await expect(summary).toContainText(/\d+ words? — \d+ points/);

  const wordCount = await page.locator(".result-word").count();
  expect(wordCount).toBeGreaterThan(0);

  // A lock fires a fire-and-forget POST to /api/telemetry (see telemetry.ts)
  // that silently no-ops wherever there's no server behind it, e.g. GitHub
  // Pages — including this plain `vite` dev server, which has no such route
  // and 404s it. Chromium logs that failed request to the console on its
  // own, unrelated to whether the app handled it; only unexpected errors
  // should fail this test.
  const unexpectedErrors = consoleErrors.filter((msg) => !/404.*Not Found/.test(msg));
  expect(unexpectedErrors, unexpectedErrors.join("\n")).toEqual([]);
});

// "Report wrong board" (scanner.ts) is the only training-submission path
// left since the manual-correction UI was stripped (see this app's
// CLAUDE.md, "UI stripped to just the camera"), and CLAUDE.md separately
// notes it went a long time genuinely untested against a live server. There
// is no submit server on the plain `vite` dev server this suite runs
// against (see submitServer.py/serveTLS.py, which are the real ones), so
// this mocks the one route it calls and checks what the client actually
// sends — the real locked photo/letters, not fixture data — rather than
// exercising the server side.
test("reports a locked board to /api/submit with the real locked photo and letters", async ({ page }) => {
  test.setTimeout(60_000);

  let submitBody: string | null = null;
  await page.route("**/api/submit", async (route) => {
    submitBody = route.request().postData();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "test-stub", wrong: true, cells: 25 }),
    });
  });

  await page.goto("/?debug=1");

  const rescanBtn = page.locator("#scanner-rescan");
  await expect(rescanBtn).toBeVisible({ timeout: 50_000 });

  const reportBtn = page.locator("#scanner-report");
  await expect(reportBtn).toBeVisible();
  await reportBtn.click();

  await expect(reportBtn).toHaveText("Reported — thanks");

  expect(submitBody).not.toBeNull();
  // Multipart body: assert the meta part actually carries a real, locked
  // 25-letter board rather than being empty or a fixed stub.
  expect(submitBody).toContain('"reportedWrong":true');
  expect(submitBody).toContain('"gridSize":5');
  const lettersMatch = submitBody!.match(/"letters":\[([^\]]*)\]/);
  expect(lettersMatch).not.toBeNull();
  const letters = JSON.parse(`[${lettersMatch![1]}]`) as string[];
  expect(letters).toHaveLength(25);
  expect(letters.every((letter) => /^[A-Z]{1,2}$/.test(letter))).toBe(true);
});
