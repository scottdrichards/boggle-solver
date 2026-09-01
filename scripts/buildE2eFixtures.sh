#!/usr/bin/env bash
# Regenerates the Chrome fake-camera video fixtures used by the Playwright
# e2e project (playwright.config.ts). These are gitignored (e2e/fixtures/*.y4m,
# see .gitignore) — large binary blobs derived from committed source images —
# so a fresh checkout needs this run once before `npm run test:e2e` can use
# the board-in-frame projects. Requires ffmpeg + python3 (stdlib only) on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="e2e/fixtures"
mkdir -p "$OUT_DIR"

# Chrome's y4m parser is stricter than ffmpeg's own muxer: a header carrying
# an XCOLORRANGE tag (which ffmpeg adds whenever the source has an explicit
# color range, as decoded JPEGs generally do) makes Chrome silently reject
# the whole file and fall back to a synthetic 2x2 black frame — discovered
# the hard way building board-clean.y4m, see the fix below. Confirmed by
# diffing against a previously-working fixture's header byte-for-byte.
strip_xcolorrange() {
  python3 - "$1" <<'PY'
import re, sys
path = sys.argv[1]
with open(path, "rb") as f:
    data = f.read()
nl = data.index(b"\n")
header = re.sub(rb" XCOLORRANGE=\w+", b"", data[:nl])
with open(path, "wb") as f:
    f.write(header + data[nl:])
PY
}

# Generic SMPTE color-bars test pattern — no board in frame. Used by the
# "chromium" project for the smoke test (app boots / camera opens / dict
# loads), which doesn't need or want a real board.
ffmpeg -y -f lavfi -i "smptebars=size=640x480:rate=10" -t 3 -pix_fmt yuv420p \
  "$OUT_DIR/fake-camera.y4m"
strip_xcolorrange "$OUT_DIR/fake-camera.y4m"

# A real 5x5 board photo (public/demo/board-working.jpg), cropped tight
# around the tray so it fills most of the frame. Used by the
# "chromium-scan-board" project for the two live scan/submit tests
# (e2e/scan-board.spec.ts). Cropped rather than fed whole because this box's
# software-rendered WebGL (no real GPU) is far slower than any device this
# project has benchmarked — the whole photo can push detection into the
# deep-pyramid fallback, which then loses a race against the scanner's own
# 15s stall watchdog under that slowdown. See playwright.config.ts.
ffmpeg -y -loop 1 -i public/demo/board-working.jpg \
  -vf "crop=1900:1500:1500:850,scale=640:480,setsar=1" \
  -t 3 -r 10 -pix_fmt yuv420p "$OUT_DIR/board-clean.y4m"
strip_xcolorrange "$OUT_DIR/board-clean.y4m"

echo "Wrote $OUT_DIR/fake-camera.y4m and $OUT_DIR/board-clean.y4m"
