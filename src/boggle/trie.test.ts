import { describe, expect, it } from "vitest";
import { Trie, letterToIndex } from "./trie";
import { loadCompiledTrie } from "./testFixtures";

function walk(trie: Trie, word: string): number {
  let node = trie.root();
  for (const ch of word) {
    node = trie.child(node, letterToIndex(ch));
    if (node === -1) break;
  }
  return node;
}

describe("compiled word trie", () => {
  const trie = loadCompiledTrie();

  it("recognizes common words", () => {
    for (const word of ["cat", "dog", "boggle", "quiz", "tree"]) {
      const node = walk(trie, word);
      expect(node, `expected "${word}" to resolve to a node`).not.toBe(-1);
      expect(trie.isWord(node), `expected "${word}" to be a word`).toBe(true);
    }
  });

  it("recognizes Q without U as its own word (qat)", () => {
    // Distinguishes the dictionary (which allows bare Q) from the Boggle
    // "Qu" die, which the solver always traverses as Q-then-U together.
    const node = walk(trie, "qat");
    expect(node).not.toBe(-1);
    expect(trie.isWord(node)).toBe(true);
  });

  it("rejects non-words and dead-end prefixes", () => {
    expect(trie.isWord(walk(trie, "zzzzz"))).toBe(false);
    expect(walk(trie, "zqxjk")).toBe(-1);
  });

  it("distinguishes a valid prefix from a complete word", () => {
    const catNode = walk(trie, "cat");
    expect(trie.isWord(catNode)).toBe(true);
    const caNode = walk(trie, "ca");
    expect(caNode).not.toBe(-1);
    expect(trie.isWord(caNode)).toBe(false);
  });
});
