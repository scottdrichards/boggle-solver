import type { Board } from "./board";
import { computeNeighbors } from "./board";
import type { Trie } from "./trie";
import { letterToIndex } from "./trie";

export interface FoundWord {
  /** Full spelled-out word, e.g. "QUIET" — note a "QU" tile contributes 2 letters. */
  word: string;
  /** Cell indices visited, in order. Shorter than `word.length` whenever a "QU" tile is used. */
  path: readonly number[];
}

/** This app's house rule is 4+ letters (no 3-letter words), stricter than
 * the trie's own 3-letter floor (see scripts/buildWordlist.ts) — enforced
 * here rather than by rebuilding the compiled dictionary. */
const MIN_WORD_LENGTH = 4;

/**
 * DFS over the board's 8-directionally-adjacent cells (each cell used at
 * most once per word), pruned by trie-prefix existence. A word is reported
 * once even if multiple paths spell it.
 */
export function solve(board: Board, trie: Trie): FoundWord[] {
  const { size, tiles } = board;
  const neighbors = computeNeighbors(size);
  const visited = new Uint8Array(tiles.length);
  const found = new Map<string, FoundWord>();
  const path: number[] = [];
  const wordChars: string[] = [];

  function visit(cell: number, trieNode: number): void {
    visited[cell] = 1;
    path.push(cell);

    let node = trieNode;
    let pushed = 0;
    for (const letter of tiles[cell]!) {
      node = trie.child(node, letterToIndex(letter));
      if (node === -1) break;
      wordChars.push(letter);
      pushed++;
    }

    if (node !== -1) {
      if (trie.isWord(node) && wordChars.length >= MIN_WORD_LENGTH) {
        const word = wordChars.join("");
        if (!found.has(word)) {
          found.set(word, { word, path: [...path] });
        }
      }
      for (const neighbor of neighbors[cell]!) {
        if (!visited[neighbor]) visit(neighbor, node);
      }
    }

    for (let i = 0; i < pushed; i++) wordChars.pop();
    path.pop();
    visited[cell] = 0;
  }

  for (let cell = 0; cell < tiles.length; cell++) {
    visit(cell, trie.root());
  }

  return [...found.values()];
}
