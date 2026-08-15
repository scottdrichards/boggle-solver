/**
 * What happens to the board *after* a lock: freezing the camera, animating
 * the photographed quad flattening into a clean square, and drawing a word's
 * path over that square when the player taps a result.
 *
 * Kept out of scanner.ts because that file's job ends at "here are 25
 * letters, confidently" — everything below is presentation, not detection.
 *
 * The flattening animation is a CSS corner-pin: the just-warped square image
 * is placed in a `<canvas>` that starts `matrix3d`-transformed so it sits
 * exactly on top of the quad as the camera saw it (indistinguishable from the
 * live board, since it *is* that board, just already de-skewed), then the
 * transform animates to identity while the frozen video fades out beneath
 * it. No per-frame re-warping is needed — `quadHomography` gives the same
 * projective mapping `warpQuadToSquare` used, just expressed as a 4x4 matrix
 * for the compositor instead of a per-pixel resample.
 *
 * The trail is drawn in the square's own grid space, which is the whole
 * reason it is drawn on the *flattened* board rather than on the live quad:
 * after dewarping, cell centres are a plain uniform grid, so connecting two
 * path steps — including diagonals — is just a straight line between two
 * fractions of the canvas, no perspective math required.
 */
import { quadHomography, type PixelBuffer, type Point } from "../cv/quadWarp";

/** A single warm "wandlight" hue used for the entire trail — tail, head and
 * sparks alike. Previously the head was a completely different hue (green
 * against the trail's blue), which read as a separate object riding on top
 * of the trail rather than the trail's own tip. Now everything is the same
 * colour family; the head is simply the brightest, most saturated point
 * (blended toward white for a "hot" glow) rather than a different colour, so
 * head and tail blend into one continuous streak. */
const TRAIL_RGB = "255, 200, 120"; // warm gold
/** The head's core is this colour blended toward white — same hue family,
 * just hotter — rather than a distinct colour, so it reads as the same
 * streak intensifying, not a separate marker. */
const TRAIL_CORE_RGB = "255, 244, 214";

/** Floor on how dim an already-visited stretch of the path gets. The path
 * stays "emphasized... just less intense" rather than fading to nothing, so
 * the whole trail already walked is still legible at a glance. */
const MIN_TRAIL_INTENSITY = 0.32;

/** How long the wand takes to glide across one letter-to-letter hop, before
 * easing. Total animation time scales with word length. */
const MS_PER_HOP = 260;
/** Catmull-Rom samples per hop. This is what removes the hard angles at each
 * letter: the spline rounds the corner instead of the path just turning on a
 * dime, and enough samples per hop makes that rounding read as a continuous
 * glide rather than a polyline. */
const SAMPLES_PER_HOP = 18;
/** The lit trail is stroked in bands rather than one gradient (a canvas
 * linear/radial gradient can't follow a curved path) or one stroke per tiny
 * sample (needlessly many draw calls) — a handful of bands, each stroked
 * once, is cheap and still reads as a smooth fade from tail to head. */
const TRAIL_BANDS = 14;

export const LOCK_TRANSITION_MS = 650;

export interface BoardLockParams {
  /** The board's quad, in the same pixel space as `frameWidth`/`frameHeight`
   * (i.e. the detect-frame buffer the fit came from), corners ordered
   * top-left, top-right, bottom-right, bottom-left. */
  readonly quad: readonly [Point, Point, Point, Point];
  readonly frameWidth: number;
  readonly frameHeight: number;
  /** The already-warped square board image (see `warpQuadToSquare`). */
  readonly board: PixelBuffer;
  readonly gridSize: number;
}

export interface BoardView {
  /** Freezes the camera view and animates the quad flattening into a clean
   * square. Safe to call again for a re-lock (a changed board): it just
   * replaces the displayed square and restarts the animation. */
  lock(params: BoardLockParams): void;
  /** Back to showing the live video feed; hides the board and any trail. */
  reset(): void;
  /** Highlights `path` (row-major cell indices, in visiting order) on the
   * flattened board, animating in step by step. `null` clears it. */
  showPath(path: readonly number[] | null): void;
  /** Paints a per-cell heatmap on the flattened board — `weights[cell]` in
   * [0,1], one entry per cell in row-major order (see `computeCellHeatmap`).
   * `null` clears it. No-op before a lock. */
  showHeatmap(weights: ArrayLike<number> | null): void;
  /** A rapid "the solver is thinking" flourish: darts a comet through each
   * of `paths` in turn, the whole sequence fit into `totalMs`. Consecutive
   * words' comets are allowed to overlap/cross-fade rather than the display
   * cutting to blank between them — see the implementation note above
   * `playFlourish`. No-op before a lock. */
  playFlourish(paths: readonly (readonly number[])[], totalMs: number): void;
  readonly locked: boolean;
}

/** Curve data for one word's path: a dense polyline plus its cumulative arc
 * length, as produced by `computeCurve`. */
interface CurveData {
  curve: Point[];
  cumDist: number[];
  totalLength: number;
}

/** Maps a point in detection-frame pixel space to CSS pixels within
 * `container`'s box, matching the `object-fit: cover` transform the live
 * `<video>` uses — the same math `drawOverlay` in scanner.ts applies to the
 * cell-centre dots, kept in sync here so the frozen quad lines up with where
 * the board actually was on screen. */
function toContainerPoint(point: Point, frameWidth: number, frameHeight: number, box: DOMRect): Point {
  const scale = Math.max(box.width / frameWidth, box.height / frameHeight);
  const offsetX = (box.width - frameWidth * scale) / 2;
  const offsetY = (box.height - frameHeight * scale) / 2;
  return { x: point.x * scale + offsetX, y: point.y * scale + offsetY };
}

/** Builds the `matrix3d(...)` that corner-pins a `boxWidth x boxHeight`
 * element (transform-origin `0 0`) onto `quad`, expressed in the same
 * coordinate frame as the element's container. This is
 * `quadHomography`/`squareToQuadMap`'s unit-square mapping, just rescaled so
 * it operates on the element's own pixel box instead of [0,1]x[0,1], and
 * written out as a column-major 4x4 matrix with the projective (w) term
 * carried in row 4 — the standard "CSS corner pin" construction. */
function quadToMatrix3d(quad: readonly [Point, Point, Point, Point], boxWidth: number, boxHeight: number): string {
  const { a11, a21, a31, a12, a22, a32, a13, a23 } = quadHomography(quad);
  // Divide the u/v-scaling terms by the box size so the matrix accepts local
  // element pixel coordinates (px, py) instead of normalized (u, v).
  const values = [
    a11 / boxWidth,
    a12 / boxWidth,
    0,
    a13 / boxWidth,
    a21 / boxHeight,
    a22 / boxHeight,
    0,
    a23 / boxHeight,
    0,
    0,
    1,
    0,
    a31,
    a32,
    0,
    1,
  ];
  return `matrix3d(${values.join(",")})`;
}

export function mountBoardView(container: HTMLElement, video: HTMLVideoElement): BoardView {
  const boardCanvas = document.createElement("canvas");
  boardCanvas.className = "scanner-board";
  boardCanvas.hidden = true;
  // Sits between the board and the trail: painted from cold cell weights up
  // to hot ones (see heatColor), independent of and layered under whatever
  // word-path trail is currently showing.
  const heatmapCanvas = document.createElement("canvas");
  heatmapCanvas.className = "scanner-heatmap";
  heatmapCanvas.hidden = true;
  const trailCanvas = document.createElement("canvas");
  trailCanvas.className = "scanner-trail";
  trailCanvas.hidden = true;
  container.append(boardCanvas, heatmapCanvas, trailCanvas);

  const boardCtx = boardCanvas.getContext("2d")!;
  const heatmapCtx = heatmapCanvas.getContext("2d")!;
  const trailCtx = trailCanvas.getContext("2d")!;

  let locked = false;
  let gridSize = 5;
  let animationFrame: number | null = null;

  /** A single spark thrown off the wand tip. Additive-blended and fading, so
   * a handful drifting away from the head is what reads as "magic" rather
   * than just a moving dot. */
  interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    born: number;
    life: number;
    size: number;
    color: string;
  }
  let particles: Particle[] = [];
  let lastFrameTime = 0;

  // The current animation's geometry, rebuilt once per showPath() call and
  // then just walked frame to frame — see buildCurve().
  let curve: Point[] = [];
  let cumDist: number[] = [];
  let totalLength = 0;
  let animStart = 0;
  let animDuration = 0;

  function clearTrail(): void {
    trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
  }

  function cellCentre(index: number, size: number): Point {
    const row = Math.floor(index / gridSize);
    const col = index % gridSize;
    return {
      x: ((col + 0.5) / gridSize) * size,
      y: ((row + 0.5) / gridSize) * size,
    };
  }

  /** Classic cold→hot ramp for a normalized weight in [0,1]: blue at 0,
   * through amber, to red at 1. Two lerps instead of an HSL conversion —
   * cheap enough to call once per cell per `showHeatmap`, and it avoids
   * HSL's washed-out midtones. */
  function heatColor(t: number): [number, number, number] {
    const lo: [number, number, number] = [64, 110, 220]; // cool blue
    const mid: [number, number, number] = [255, 210, 60]; // amber
    const hi: [number, number, number] = [230, 40, 40]; // hot red
    const [a, b] = t < 0.5 ? [lo, mid] : [mid, hi];
    const u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  }

  function clearHeatmap(): void {
    heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
  }

  // Hard-edged cell rectangles are painted onto this offscreen buffer first,
  // then the whole buffer is composited onto the visible canvas through a
  // blur filter (see drawHeatmap). That turns a checkerboard of flat blocks
  // into soft, overlapping blobs — the "feathered" heatmap look — in one
  // filtered drawImage call, rather than hand-blending a radial gradient per
  // cell. Kept as persistent module state instead of a fresh canvas per call
  // since a redraw happens on every toggle/result change.
  const heatmapBuffer = document.createElement("canvas");
  const heatmapBufferCtx = heatmapBuffer.getContext("2d")!;

  function drawHeatmap(weights: ArrayLike<number>): void {
    const size = boardCanvas.width;
    heatmapCanvas.width = size;
    heatmapCanvas.height = boardCanvas.height;
    heatmapBuffer.width = size;
    heatmapBuffer.height = boardCanvas.height;
    clearHeatmap();
    heatmapBufferCtx.clearRect(0, 0, heatmapBuffer.width, heatmapBuffer.height);

    const cell = size / gridSize;
    for (let i = 0; i < weights.length; i++) {
      const t = Math.max(0, Math.min(1, weights[i]!));
      const row = Math.floor(i / gridSize);
      const col = i % gridSize;
      const [r, g, b] = heatColor(t);
      // Cold cells fade to nothing rather than sitting as a flat blue wash,
      // so an all-zero board (nothing found yet) shows no tint at all.
      const alpha = 0.14 + 0.68 * t;
      heatmapBufferCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      heatmapBufferCtx.fillRect(col * cell, row * cell, cell, cell);
    }

    // Blur radius scales with cell size so feathering looks the same
    // proportionally regardless of board resolution — about a third of a
    // cell bleeds into its neighbours, enough to round the grid lines away
    // without smearing distinct hot/cold cells into one indistinct wash.
    heatmapCtx.filter = `blur(${(cell * 0.35).toFixed(1)}px)`;
    heatmapCtx.drawImage(heatmapBuffer, 0, 0);
    heatmapCtx.filter = "none";
  }

  function showHeatmap(weights: ArrayLike<number> | null): void {
    if (!weights || weights.length === 0 || !locked) {
      clearHeatmap();
      heatmapCanvas.hidden = true;
      return;
    }
    heatmapCanvas.hidden = false;
    drawHeatmap(weights);
  }

  /** Standard uniform Catmull-Rom: passes exactly through p1 and p2 at
   * t=0/1, and its tangents there are set by the neighbouring points, which
   * is what rounds the corner at a letter instead of the path kinking
   * through it. */
  function catmullRom(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
    const t2 = t * t;
    const t3 = t2 * t;
    return {
      x:
        0.5 *
        (2 * p1.x +
          (p2.x - p0.x) * t +
          (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
          (3 * p1.x - p0.x - 3 * p2.x + p3.x) * t3),
      y:
        0.5 *
        (2 * p1.y +
          (p2.y - p0.y) * t +
          (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
          (3 * p1.y - p0.y - 3 * p2.y + p3.y) * t3),
    };
  }

  /** Turns a path's cell centres into a dense, smooth polyline plus its
   * cumulative arc length, so the wand can move along it at a constant
   * on-screen speed regardless of how far apart two letters happen to sit.
   * Endpoints are duplicated so the spline has a tangent to work with at the
   * very first and last letter too. Pure (returns data rather than mutating
   * module state) so it can be reused by both the normal per-word glide and
   * the multi-word thinking flourish below without the two stepping on each
   * other's in-flight animation state. */
  function computeCurve(path: readonly number[], size: number): CurveData {
    const points = path.map((index) => cellCentre(index, size));
    const padded = [points[0]!, ...points, points[points.length - 1]!];

    const samples: Point[] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = padded[i]!;
      const p1 = padded[i + 1]!;
      const p2 = padded[i + 2]!;
      const p3 = padded[i + 3]!;
      for (let s = 0; s < SAMPLES_PER_HOP; s++) {
        samples.push(catmullRom(p0, p1, p2, p3, s / SAMPLES_PER_HOP));
      }
    }
    samples.push(points[points.length - 1]!);

    const cumDist = [0];
    for (let i = 1; i < samples.length; i++) {
      const a = samples[i - 1]!;
      const b = samples[i]!;
      cumDist.push(cumDist[i - 1]! + Math.hypot(b.x - a.x, b.y - a.y));
    }
    return { curve: samples, cumDist, totalLength: cumDist[cumDist.length - 1] ?? 0 };
  }

  function buildCurve(path: readonly number[], size: number): void {
    const data = computeCurve(path, size);
    curve = data.curve;
    cumDist = data.cumDist;
    totalLength = data.totalLength;
  }

  /** The point `distance` along `data.curve`, plus the index of the curve
   * sample just behind it (so the caller can stroke "everything up to
   * here"). */
  function pointAlong(data: CurveData, distance: number): { point: Point; index: number } {
    if (data.curve.length === 0) return { point: { x: 0, y: 0 }, index: 0 };
    const target = Math.max(0, Math.min(data.totalLength, distance));
    let index = 0;
    while (index < data.cumDist.length - 1 && data.cumDist[index + 1]! < target) index++;
    const a = data.curve[index]!;
    const b = data.curve[Math.min(index + 1, data.curve.length - 1)]!;
    const segStart = data.cumDist[index]!;
    const segLen = data.cumDist[Math.min(index + 1, data.cumDist.length - 1)]! - segStart || 1;
    const localT = (target - segStart) / segLen;
    return { point: { x: a.x + (b.x - a.x) * localT, y: a.y + (b.y - a.y) * localT }, index };
  }

  /** The point `distance` along the current single-word `curve` — a thin
   * wrapper over `pointAlong` kept for `renderFrame`, which tracks the
   * in-progress word as module state rather than a `CurveData` value. */
  function pointAtDistance(distance: number): { point: Point; index: number } {
    return pointAlong({ curve, cumDist, totalLength }, distance);
  }


  /** Ease-out-cubic: the wand is already at full speed the instant a word is
   * tapped — no slow ramp-up off the first letter — and only decelerates
   * approaching the last one, so it still settles into place rather than
   * snapping to a stop. (Previously ease-in-out-cubic, which added a slow
   * start before the wand had visibly moved at all.) */
  function easeOutCubic(f: number): number {
    return 1 - Math.pow(1 - f, 3);
  }

  function spawnParticle(at: Point): void {
    const angle = Math.random() * Math.PI * 2;
    const speed = 0.35 + Math.random() * 0.9;
    // Same hue family as the trail (mostly the hot core, occasionally the
    // base gold) so sparks read as flecks of the trail's own light, not a
    // separately-coloured effect layered on top.
    const color = Math.random() < 0.6 ? TRAIL_CORE_RGB : TRAIL_RGB;
    particles.push({
      x: at.x,
      y: at.y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      born: performance.now(),
      life: 450 + Math.random() * 350,
      size: 2 + Math.random() * 2.5,
      color,
    });
  }

  function updateAndDrawParticles(now: number, dtMs: number): void {
    particles = particles.filter((particle) => now - particle.born < particle.life);
    trailCtx.save();
    trailCtx.globalCompositeOperation = "lighter";
    for (const particle of particles) {
      particle.x += particle.vx * (dtMs / 16);
      particle.y += particle.vy * (dtMs / 16);
      const age = (now - particle.born) / particle.life;
      const alpha = (1 - age) * 0.85;
      const size = particle.size * (1 - age * 0.7);
      trailCtx.beginPath();
      trailCtx.fillStyle = `rgba(${particle.color}, ${Math.max(0, alpha)})`;
      trailCtx.arc(particle.x, particle.y, Math.max(0, size), 0, Math.PI * 2);
      trailCtx.fill();
    }
    trailCtx.restore();
  }

  /** One animation frame: strokes the already-travelled curve in fading
   * bands, draws the glowing (semi-transparent, so the letters underneath
   * stay visible) wand tip, and steps the particle trail it throws off. */
  function renderFrame(now: number): void {
    const f = animDuration <= 0 ? 1 : Math.min(1, (now - animStart) / animDuration);
    const eased = easeOutCubic(f);
    const headDistance = totalLength * eased;
    const { point: head, index: headIndex } = pointAtDistance(headDistance);

    clearTrail();

    // Band the travelled portion of the curve from tail (dim) to head
    // (bright) — a handful of strokes rather than one per sample, which
    // stays smooth without a draw call per sample point.
    const visibleCount = headIndex + 1;
    if (visibleCount > 1 || headDistance > 0) {
      const bandSize = Math.max(1, Math.ceil(visibleCount / TRAIL_BANDS));
      for (let start = 0; start < visibleCount; start += bandSize) {
        const end = Math.min(visibleCount, start + bandSize + 1); // +1 overlap so bands join
        const bandPoints = curve.slice(start, end);
        if (start + bandSize >= visibleCount) bandPoints.push(head);
        if (bandPoints.length < 2) continue;

        const recencyFraction = start / Math.max(1, visibleCount - 1);
        const intensity = MIN_TRAIL_INTENSITY + (1 - MIN_TRAIL_INTENSITY) * recencyFraction;

        trailCtx.beginPath();
        trailCtx.moveTo(bandPoints[0]!.x, bandPoints[0]!.y);
        for (const point of bandPoints.slice(1)) trailCtx.lineTo(point.x, point.y);
        trailCtx.strokeStyle = `rgba(${TRAIL_RGB}, ${intensity})`;
        trailCtx.lineWidth = Math.max(3, trailCanvas.width / gridSize / 7);
        trailCtx.lineCap = "round";
        trailCtx.lineJoin = "round";
        trailCtx.stroke();
      }
    }

    // Spawn sparks only while the wand is still moving — a finished trail
    // should settle, not keep throwing off particles forever.
    if (f < 1) {
      spawnParticle(head);
    }
    const dtMs = lastFrameTime === 0 ? 16 : now - lastFrameTime;
    lastFrameTime = now;
    updateAndDrawParticles(now, dtMs);

    // The wand tip: the *same* hue as the trail, just blended toward white
    // and drawn with additive blending, so it reads as the trail's own light
    // intensifying at the tip rather than a differently-coloured marker
    // riding on top of it. No hard-edged ring around it, either — the whole
    // thing is one soft gradient fading to nothing, so there is no boundary
    // where "head" visibly ends and "tail" begins.
    const cellSize = trailCanvas.width / gridSize;
    trailCtx.save();
    trailCtx.globalCompositeOperation = "lighter";
    const glow = trailCtx.createRadialGradient(head.x, head.y, 0, head.x, head.y, cellSize * 0.6);
    glow.addColorStop(0, `rgba(${TRAIL_CORE_RGB}, 0.75)`);
    glow.addColorStop(0.4, `rgba(${TRAIL_RGB}, 0.45)`);
    glow.addColorStop(1, `rgba(${TRAIL_RGB}, 0)`);
    trailCtx.fillStyle = glow;
    trailCtx.beginPath();
    trailCtx.arc(head.x, head.y, cellSize * 0.6, 0, Math.PI * 2);
    trailCtx.fill();
    trailCtx.restore();

    if (f < 1 || particles.length > 0) {
      animationFrame = requestAnimationFrame(renderFrame);
    } else {
      animationFrame = null;
    }
  }

  function stopAnimation(): void {
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    particles = [];
    lastFrameTime = 0;
  }

  function showPath(path: readonly number[] | null): void {
    stopAnimation();
    if (!path || path.length === 0 || !locked) {
      clearTrail();
      return;
    }

    // Size the trail canvas to match the board canvas's own resolution so
    // cell fractions line up pixel-for-pixel with the letters beneath it.
    trailCanvas.width = boardCanvas.width;
    trailCanvas.height = boardCanvas.height;

    buildCurve(path, trailCanvas.width);
    animStart = performance.now();
    // Total glide time scales with the number of hops, so a two-letter word
    // doesn't linger and a long one isn't a blur.
    animDuration = Math.max(1, path.length - 1) * MS_PER_HOP;
    animationFrame = requestAnimationFrame(renderFrame);
  }

  /** Darts a short comet through each word in `paths`, sized to fit the
   * whole sequence into `totalMs`. This is deliberately a separate loop from
   * `renderFrame`/`showPath` rather than just calling `showPath` on a timer:
   * `showPath` always starts its word's glide at progress 0, so cutting it
   * short every ~150ms (10 words in ~1.5s) just kept re-flashing the first
   * letter of each new word — it never got anywhere near the second one.
   * Here a single rAF loop owns the whole sequence and always draws the
   * comet at its *current* position along whichever word is "up" for the
   * elapsed time, so it visibly darts from letter to letter and from word to
   * word instead of blinking in place.
   *
   * The trail canvas is faded each frame (`destination-out`) rather than
   * hard-cleared, so one word's tail is still fading out additively when
   * the next word's comet starts — overlapping trails read as fast,
   * continuous motion, which is the effect asked for; a hard clear between
   * words is what produced the "blinking" complaint. */
  function playFlourish(paths: readonly (readonly number[])[], totalMs: number): void {
    stopAnimation();
    if (!locked) return;

    const size = boardCanvas.width;
    const words = paths.filter((path) => path.length > 1).map((path) => computeCurve(path, size));
    if (words.length === 0) return;

    trailCanvas.width = boardCanvas.width;
    trailCanvas.height = boardCanvas.height;
    clearTrail();

    // Each word gets a slice of totalMs proportional to its own arc length
    // (with a floor), so the comet moves at roughly one constant speed
    // across the whole flourish instead of longer words racing past faster
    // than short ones.
    const MIN_SEGMENT_MS = 70;
    const arcLengths = words.map((word) => Math.max(word.totalLength, 1));
    const totalArc = arcLengths.reduce((sum, length) => sum + length, 0);
    const rawSegmentMs = arcLengths.map((length) => Math.max(MIN_SEGMENT_MS, (length / totalArc) * totalMs));
    const rawTotal = rawSegmentMs.reduce((sum, ms) => sum + ms, 0);
    const scale = rawTotal > 0 ? totalMs / rawTotal : 1;
    const segmentMs = rawSegmentMs.map((ms) => ms * scale);

    const start = performance.now();

    function frame(now: number): void {
      const elapsed = now - start;

      trailCtx.save();
      trailCtx.globalCompositeOperation = "destination-out";
      trailCtx.fillStyle = "rgba(0, 0, 0, 0.3)";
      trailCtx.fillRect(0, 0, trailCanvas.width, trailCanvas.height);
      trailCtx.restore();

      let remaining = elapsed;
      let wordIndex = 0;
      while (wordIndex < segmentMs.length && remaining >= segmentMs[wordIndex]!) {
        remaining -= segmentMs[wordIndex]!;
        wordIndex++;
      }

      if (wordIndex >= words.length) {
        clearTrail();
        animationFrame = null;
        return;
      }

      const data = words[wordIndex]!;
      const duration = segmentMs[wordIndex]!;
      const f = duration <= 0 ? 1 : Math.min(1, remaining / duration);
      const headDistance = data.totalLength * f;
      const { point: head, index: headIndex } = pointAlong(data, headDistance);

      // A short comet tail behind the head, same warm hue as the normal
      // trail, additively blended so overlapping comets brighten rather
      // than visually fight each other.
      const tailStart = Math.max(0, headIndex - Math.round(SAMPLES_PER_HOP * 1.5));
      const tailPoints = [...data.curve.slice(tailStart, headIndex + 1), head];
      if (tailPoints.length > 1) {
        trailCtx.save();
        trailCtx.globalCompositeOperation = "lighter";
        trailCtx.beginPath();
        trailCtx.moveTo(tailPoints[0]!.x, tailPoints[0]!.y);
        for (const point of tailPoints.slice(1)) trailCtx.lineTo(point.x, point.y);
        trailCtx.strokeStyle = `rgba(${TRAIL_RGB}, 0.85)`;
        trailCtx.lineWidth = Math.max(3, trailCanvas.width / gridSize / 6);
        trailCtx.lineCap = "round";
        trailCtx.lineJoin = "round";
        trailCtx.stroke();
        trailCtx.restore();
      }

      const cellSize = trailCanvas.width / gridSize;
      trailCtx.save();
      trailCtx.globalCompositeOperation = "lighter";
      const glow = trailCtx.createRadialGradient(head.x, head.y, 0, head.x, head.y, cellSize * 0.55);
      glow.addColorStop(0, `rgba(${TRAIL_CORE_RGB}, 0.9)`);
      glow.addColorStop(0.4, `rgba(${TRAIL_RGB}, 0.5)`);
      glow.addColorStop(1, `rgba(${TRAIL_RGB}, 0)`);
      trailCtx.fillStyle = glow;
      trailCtx.beginPath();
      trailCtx.arc(head.x, head.y, cellSize * 0.55, 0, Math.PI * 2);
      trailCtx.fill();
      trailCtx.restore();

      animationFrame = requestAnimationFrame(frame);
    }

    animationFrame = requestAnimationFrame(frame);
  }

  function lock(params: BoardLockParams): void {
    stopAnimation();
    gridSize = params.gridSize;
    const box = video.getBoundingClientRect();
    const containerQuad = params.quad.map((point) =>
      toContainerPoint(point, params.frameWidth, params.frameHeight, box),
    ) as [Point, Point, Point, Point];

    boardCanvas.width = params.board.width;
    boardCanvas.height = params.board.height;
    boardCtx.putImageData(
      new ImageData(new Uint8ClampedArray(params.board.data), params.board.width, params.board.height),
      0,
      0,
    );

    boardCanvas.hidden = false;
    trailCanvas.hidden = false;
    trailCanvas.width = params.board.width;
    trailCanvas.height = params.board.height;
    clearTrail();

    // Start pinned exactly over the quad — at this instant the board layer
    // is visually indistinguishable from the live frame beneath it, since it
    // shows the same board, merely already de-skewed.
    boardCanvas.style.transition = "none";
    boardCanvas.style.opacity = "1";
    boardCanvas.style.transform = quadToMatrix3d(containerQuad, box.width, box.height);
    video.style.transition = "none";
    video.style.opacity = "1";
    // Force a style flush so the next assignment is seen as a change to
    // transition, not folded into the one above.
    void boardCanvas.offsetWidth;

    boardCanvas.style.transition = `transform ${LOCK_TRANSITION_MS}ms cubic-bezier(0.22, 0.68, 0.24, 1)`;
    video.style.transition = `opacity ${LOCK_TRANSITION_MS}ms ease`;
    requestAnimationFrame(() => {
      boardCanvas.style.transform = "none";
      video.style.opacity = "0";
    });

    locked = true;
  }

  function reset(): void {
    stopAnimation();
    locked = false;
    boardCanvas.hidden = true;
    trailCanvas.hidden = true;
    heatmapCanvas.hidden = true;
    clearTrail();
    clearHeatmap();
    boardCanvas.style.transition = "none";
    boardCanvas.style.transform = "none";
    video.style.transition = "none";
    video.style.opacity = "1";
  }

  return {
    lock,
    reset,
    showPath,
    showHeatmap,
    playFlourish,
    get locked() {
      return locked;
    },
  };
}
