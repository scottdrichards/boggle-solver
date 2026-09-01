import { SolverClient } from "../workers/solverClient";
import { computeCellHeatmap } from "../boggle/heatmap";
import type { ScoredWord } from "../boggle/score";
import { renderResults, updateHighlightedWord } from "./results";
import { mountScanner } from "./scanner";
import { minWordLengthFor, setHighlightedWord, setResults, state } from "./state";

/** How many words flash during the post-lock "thinking" flourish, and the
 * total wall-clock time it's allotted (see playThinkingFlourish). */
const THINKING_FLOURISH_WORD_COUNT = 10;
const THINKING_FLOURISH_TOTAL_MS = 1500;

export async function mountApp(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <header class="app-header">
      <h1>Boggle Solver</h1>
      <p class="dictionary-status" id="dictionary-status" hidden></p>
    </header>
    <main class="board-and-results">
      <section class="scanner-panel" id="scanner-panel"></section>
      <div class="results" id="results"></div>
    </main>
  `;

  const resultsEl = root.querySelector<HTMLElement>("#results")!;
  const scannerPanel = root.querySelector<HTMLElement>("#scanner-panel")!;
  const dictionaryStatusEl = root.querySelector<HTMLElement>("#dictionary-status")!;

  const solver = new SolverClient();
  let dictionaryReady = false;
  let heatmapOn = false;

  /** The scanner's whole point is that finding the board and solving it are
   * one uninterrupted motion. A lock now freezes the camera and flattens the
   * board on screen (see scanner.ts/boardView.ts), so results appear over a
   * still, readable board rather than a live feed — "Scan a new board"
   * hands control back to the camera. */
  const scanner = mountScanner(scannerPanel, {
    onLocked() {
      void runSolve();
    },
  });

  /** Recomputes and (re)draws the per-cell point heatmap from the current
   * results, or clears it — called both on toggle and whenever the results
   * set changes while the toggle is already on, since a new board means the
   * old weights no longer apply to what's on screen. */
  function applyHeatmap(): void {
    scanner.showHeatmap(heatmapOn ? computeCellHeatmap(state.results, state.tiles.length) : null);
  }

  function refreshResults(): void {
    renderResults(resultsEl, state.results, state.totalPoints, state.highlightedWord, heatmapOn, {
      onSelectWord(word) {
        setHighlightedWord(word);
        const scored = word ? state.results.find((result) => result.word === word) : undefined;
        scanner.showPath(scored ? scored.path : null);
        updateHighlightedWord(resultsEl, word);
      },
      onToggleHeatmap() {
        heatmapOn = !heatmapOn;
        applyHeatmap();
        refreshResults();
      },
    });
  }

  refreshResults();

  /** Evenly samples up to `count` words by length (longest to shortest) so
   * the flourish shows a variety of word lengths rather than whatever
   * happened to sort first. */
  function sampleForFlourish(words: readonly ScoredWord[], count: number): ScoredWord[] {
    if (words.length <= count) return [...words];
    const byLength = [...words].sort((a, b) => b.word.length - a.word.length);
    const step = byLength.length / count;
    return Array.from({ length: count }, (_, i) => byLength[Math.floor(i * step)]!);
  }

  /** A quick "the solver is thinking" flourish: darts the wand trail through
   * a handful of found words right after a lock, purely cosmetic — real
   * word selection still works normally once it's done. See
   * `BoardView.playFlourish` for why this is a single continuous animation
   * rather than repeated `showPath` calls on a timer (the latter just
   * blinked at each word's first letter). */
  function playThinkingFlourish(words: readonly ScoredWord[]): void {
    const sample = sampleForFlourish(words, THINKING_FLOURISH_WORD_COUNT);
    if (sample.length === 0) return;
    scanner.playFlourish(
      sample.map((word) => word.path),
      THINKING_FLOURISH_TOTAL_MS,
    );
  }

  async function runSolve(): Promise<void> {
    if (!dictionaryReady || state.tiles.some((tile) => tile === "")) return;
    const outcome = await solver.solve(state.tiles, minWordLengthFor(state.gridSize));
    setResults(outcome.words, outcome.totalPoints);
    refreshResults();
    applyHeatmap();
    playThinkingFlourish(state.results);
  }

  try {
    await solver.init(`${import.meta.env.BASE_URL}wordlist.trie.bin`);
    dictionaryReady = await solver.dictionaryReady;
    if (!dictionaryReady) throw new Error("worker reported dictionary parse failure");
  } catch (error) {
    dictionaryStatusEl.textContent = "Couldn't load the dictionary — try reloading the page.";
    dictionaryStatusEl.classList.add("error");
    dictionaryStatusEl.hidden = false;
    console.error(error);
  }

  // The app's whole interaction: open it and point it at a board. The camera
  // starts on its own — no tap required — so this is the entire golden path.
  void scanner.start();

  // scanner.ts owns its own pipeline worker's pagehide teardown (a longer
  // story — see PIPELINE_IDLE_MS there); the solver worker just needs this.
  window.addEventListener("pagehide", () => solver.terminate());
}
