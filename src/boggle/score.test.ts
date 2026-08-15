import { describe, expect, it } from "vitest";
import { classicScore, scoreWords, totalScore } from "./score";
import type { FoundWord } from "./solve";

describe("classicScore", () => {
  it.each([
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 3],
    [7, 5],
    [8, 11],
    [12, 11],
  ])("scores a %i-letter word as %i points", (length, expected) => {
    expect(classicScore("A".repeat(length))).toBe(expected);
  });

  it("counts QUIZ as 4 letters (1 point), not 3 board cells", () => {
    // Mirrors the solver: a "QU" tile contributes 2 letters to word.length.
    expect(classicScore("QUIZ")).toBe(1);
  });
});

describe("scoreWords / totalScore", () => {
  const found: FoundWord[] = [
    { word: "CAT", path: [0, 1, 2] },
    { word: "QUIZ", path: [3, 4, 5] },
  ];

  it("scores with the classic strategy and sums correctly", () => {
    const scored = scoreWords(found);
    expect(scored.map((w) => w.points)).toEqual([1, 1]);
    expect(totalScore(scored)).toBe(2);
  });

  it("preserves the original word and path alongside points", () => {
    const [cat] = scoreWords(found);
    expect(cat).toMatchObject({ word: "CAT", path: [0, 1, 2], points: 1 });
  });
});
