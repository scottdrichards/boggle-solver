const MAGIC = 0x31_4c_47_42; // "BGL1" little-endian, must match scripts/buildWordlist.ts
const IS_WORD_BIT = 26;
const LETTER_MASK = (1 << 26) - 1; // bits 0-25

function popcount(x: number): number {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >> 24;
}

/** 'a'-'z' (or 'A'-'Z') -> 0-25. */
export function letterToIndex(letter: string): number {
  return letter.toLowerCase().charCodeAt(0) - 97;
}

/**
 * Read-only view over a compiled word DAWG (see scripts/buildWordlist.ts for
 * the binary format and why a trie was minimized into a DAG). Traversal is
 * one letter-index at a time so callers decide how a grid cell maps to
 * letters — e.g. a "Qu" tile is two calls, `child(node, Q)` then
 * `child(node, U)`, not a special case in the trie itself.
 */
export class Trie {
  private constructor(
    private readonly rootId: number,
    private readonly nodeMask: Uint32Array,
    private readonly nodeChildStart: Uint32Array,
    private readonly childIndices: Uint32Array,
  ) {}

  static async load(url: string): Promise<Trie> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`failed to load trie from ${url}: ${response.status}`);
    }
    return Trie.fromBuffer(await response.arrayBuffer());
  }

  static fromBuffer(buffer: ArrayBuffer): Trie {
    const header = new Uint32Array(buffer, 0, 4);
    const [magic, rootId, nodeCount, edgeCount] = header as unknown as [number, number, number, number];
    if (magic !== MAGIC) {
      throw new Error("not a valid boggle-solver trie file");
    }

    let offset = header.byteLength;
    const nodeMask = new Uint32Array(buffer, offset, nodeCount);
    offset += nodeMask.byteLength;
    const nodeChildStart = new Uint32Array(buffer, offset, nodeCount);
    offset += nodeChildStart.byteLength;
    const childIndices = new Uint32Array(buffer, offset, edgeCount);

    return new Trie(rootId, nodeMask, nodeChildStart, childIndices);
  }

  root(): number {
    return this.rootId;
  }

  isWord(nodeId: number): boolean {
    return (this.nodeMask[nodeId]! & (1 << IS_WORD_BIT)) !== 0;
  }

  /** Next node for `letterIndex` (0-25) from `nodeId`, or -1 if no word has this prefix. */
  child(nodeId: number, letterIndex: number): number {
    const mask = this.nodeMask[nodeId]!;
    const bit = 1 << letterIndex;
    if ((mask & bit) === 0) return -1;

    const precedingBits = mask & LETTER_MASK & (bit - 1);
    const position = popcount(precedingBits);
    return this.childIndices[this.nodeChildStart[nodeId]! + position]!;
  }
}
