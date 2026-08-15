import { describe, expect, it } from "vitest";
import { computeNeighbors, createBoard } from "./board";
import { solve } from "./solve";
import { loadCompiledTrie } from "./testFixtures";
import { letterToIndex } from "./trie";

// 3x3 board (the solver is grid-size-agnostic):
//   C  A  T
//  QU  I  Z
//   X  X  X
// Indices:    0  1  2
//             3  4  5
//             6  7  8
const board = createBoard(["C", "A", "T", "QU", "I", "Z", "X", "X", "X"]);
const trie = loadCompiledTrie();

describe("solve", () => {
  const results = solve(board, trie);
  const byWord = new Map(results.map((r) => [r.word, r]));

  it("excludes 3-letter words (this app's house rule, stricter than the trie's own 3-letter floor)", () => {
    expect(byWord.has("CAT")).toBe(false);
  });

  it("finds a simple straight-line word", () => {
    const quit = byWord.get("QUIT");
    expect(quit).toBeDefined();
    expect(quit!.path).toEqual([3, 4, 2]);
  });

  it("treats a QU tile as two letters, but one path cell", () => {
    const quiz = byWord.get("QUIZ");
    expect(quiz).toBeDefined();
    expect(quiz!.word).toBe("QUIZ");
    expect(quiz!.path).toEqual([3, 4, 5]);
    expect(quiz!.path.length).toBeLessThan(quiz!.word.length);
  });

  it("never reuses a cell within a word", () => {
    // "TAT" would require revisiting the board's single T (cell 2).
    expect(byWord.has("TAT")).toBe(false);
  });

  it("only chains through real 8-directional neighbors", () => {
    const neighbors = computeNeighbors(board.size);
    for (const { path } of results) {
      for (let i = 1; i < path.length; i++) {
        const prev = path[i - 1]!;
        const cur = path[i]!;
        expect(neighbors[prev]).toContain(cur);
      }
    }
  });

  it("reports every word at most once", () => {
    const words = results.map((r) => r.word);
    expect(new Set(words).size).toBe(words.length);
  });

  it("only reports words the trie actually recognizes", () => {
    expect(results.length).toBeGreaterThan(0);
    for (const { word } of results) {
      let node = trie.root();
      for (const ch of word) node = trie.child(node, letterToIndex(ch));
      expect(node, `"${word}" should resolve to a trie node`).not.toBe(-1);
      expect(trie.isWord(node), `"${word}" should be marked as a word`).toBe(true);
    }
  });
});
