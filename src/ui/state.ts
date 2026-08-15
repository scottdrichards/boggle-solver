import type { ScoredWord } from "../boggle/score";

/** 5x5 (Big Boggle) only for now — the 4x4 path is disabled app-wide while we
 * concentrate the model on the one board that actually gets played here. The
 * lattice fitter is also pinned to 5 so it can never fit a 4x4 sub-lattice on
 * a 5x5 board, which would mis-slice every cell. */
export type GridSize = 5;

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
