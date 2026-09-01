import type { ScoredWord } from "../boggle/score";

/** 4x4 (classic Boggle) or 5x5 (Big Boggle) — the scanner auto-detects which
 * one it's looking at (`src/cv/latticeFit.ts` tries both, tie-broken toward
 * the smaller board, and only reads 5 when the extra ring of peaks is
 * actually there). `state.gridSize` tracks whatever the current scan last
 * detected; it's a starting guess until a board actually locks. */
export type GridSize = 4 | 5;

/** Classic 4x4 sets use a 3-letter minimum; Big Boggle (5x5) uses 4 — this
 * app's existing house rule. */
export function minWordLengthFor(gridSize: GridSize): number {
  return gridSize === 4 ? 3 : 4;
}

export interface AppState {
  gridSize: GridSize;
  tiles: string[];
  results: ScoredWord[];
  totalPoints: number;
  highlightedWord: string | null;
}

function emptyTiles(gridSize: GridSize): string[] {
  return Array.from({ length: gridSize * gridSize }, () => "");
}

export const GRID_SIZE: GridSize = 5;

export const state: AppState = {
  gridSize: GRID_SIZE,
  tiles: emptyTiles(GRID_SIZE),
  results: [],
  totalPoints: 0,
  highlightedWord: null,
};

export function setGridSize(gridSize: GridSize): void {
  state.gridSize = gridSize;
}

export function setTiles(values: readonly string[]): void {
  state.tiles = [...values];
}

export function setResults(results: ScoredWord[], totalPoints: number): void {
  state.results = results;
  state.totalPoints = totalPoints;
  state.highlightedWord = null;
}

export function setHighlightedWord(word: string | null): void {
  state.highlightedWord = word;
}
