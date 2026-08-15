import type { FoundWord } from "./solve";

export interface ScoredWord extends FoundWord {
  points: number;
}

// Standard Boggle: word length is counted in letters, so a "QU" tile is
// worth 2 toward length even though it's a single board cell.
const CLASSIC_LENGTH_POINTS: Readonly<Record<number, number>> = { 3: 1, 4: 1, 5: 2, 6: 3, 7: 5 };
const CLASSIC_LONG_WORD_POINTS = 11; // 8+ letters

export function classicScore(word: string): number {
  if (word.length >= 8) return CLASSIC_LONG_WORD_POINTS;
  return CLASSIC_LENGTH_POINTS[word.length] ?? 0;
}

export function scoreWords(found: readonly FoundWord[]): ScoredWord[] {
  return found.map((word) => ({ ...word, points: classicScore(word.word) }));
}

export function totalScore(scored: readonly ScoredWord[]): number {
  return scored.reduce((sum, word) => sum + word.points, 0);
}
