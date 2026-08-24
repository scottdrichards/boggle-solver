import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

// NASPA Word List 2023 (NWL2023) headword list (one word per line, no
// definitions) — the current North American tournament Scrabble word list,
// successor to the Official Scrabble Players Dictionary (OSPD). Every valid
// word, inflected forms included as their own entries. Extracted from
// scrabblewords/scrabblewords' words/North-American/NWL2023.txt by taking
// just the leading token off each entry.
const WORDLIST_PATH = resolve("scripts/wordlists/nwl2023.txt");
const words = readFileSync(WORDLIST_PATH, "utf8").split("\n").filter(Boolean);

// Binary DAWG format ("BGL1"), read by src/boggle/trie.ts:
//   uint32 magic, uint32 rootId, uint32 nodeCount, uint32 edgeCount
//   uint32[nodeCount]  nodeMask       (bits 0-25: has-child-for-letter a-z; bit 26: isWord)
//   uint32[nodeCount]  nodeChildStart (start offset into childIndices, in bitmask/alphabetical order)
//   uint32[edgeCount]  childIndices   (child node id for each set bit, across all nodes)
//
// A plain (unminimized) trie over this word list serializes to >7MB, mostly
// because a trie never shares suffixes. Since English has enormous suffix
// overlap ("-ing", "-tion", "-ed", plurals...), we minimize the trie into a
// DAWG (directed acyclic word graph) by merging nodes with identical
// (isWord, children) signatures, bottom-up. This regularly gets dictionaries
// like this one down to a fraction of the plain-trie size.
const MAGIC = 0x31_4c_47_42; // "BGL1" little-endian
const MIN_WORD_LEN = 3;
const MAX_WORD_LEN = 20;
const IS_WORD_BIT = 26;

interface BuildNode {
  children: Map<number, BuildNode>;
  isWord: boolean;
}

function letterIndex(ch: string): number {
  return ch.charCodeAt(0) - "a".charCodeAt(0);
}

function buildTrie(wordList: readonly string[]): { root: BuildNode; wordCount: number } {
  const root: BuildNode = { children: new Map(), isWord: false };
  let wordCount = 0;

  for (const raw of wordList) {
    const word = raw.toLowerCase();
    if (word.length < MIN_WORD_LEN || word.length > MAX_WORD_LEN) continue;
    if (!/^[a-z]+$/.test(word)) continue;

    let node = root;
    for (const ch of word) {
      const li = letterIndex(ch);
      let next = node.children.get(li);
      if (!next) {
        next = { children: new Map(), isWord: false };
        node.children.set(li, next);
      }
      node = next;
    }
    node.isWord = true;
    wordCount++;
  }

  return { root, wordCount };
}

interface MinimizedGraph {
  rootId: number;
  nodeMask: Uint32Array;
  /** Minimized child id per node, per letter present in that node's mask, in ascending letter order. */
  nodeChildren: number[][];
}

/** Bottom-up trie -> DAWG minimization via structural-equality signatures. */
function minimize(root: BuildNode): MinimizedGraph {
  const signatureToId = new Map<string, number>();
  const nodeMask: number[] = [];
  const nodeChildren: number[][] = [];

  function visit(node: BuildNode): number {
    const sortedKeys = [...node.children.keys()].sort((a, b) => a - b);
    const childIds = sortedKeys.map((key) => visit(node.children.get(key)!));

    const signature = `${node.isWord ? 1 : 0}|${sortedKeys.map((key, i) => `${key}:${childIds[i]}`).join(",")}`;
    const existingId = signatureToId.get(signature);
    if (existingId !== undefined) return existingId;

    let mask = node.isWord ? 1 << IS_WORD_BIT : 0;
    for (const key of sortedKeys) mask |= 1 << key;

    const id = nodeMask.length;
    nodeMask.push(mask >>> 0);
    nodeChildren.push(childIds);
    signatureToId.set(signature, id);
    return id;
  }

  const rootId = visit(root);
  return { rootId, nodeMask: Uint32Array.from(nodeMask), nodeChildren };
}

function serialize(graph: MinimizedGraph): Buffer {
  const nodeCount = graph.nodeMask.length;
  const nodeChildStart = new Uint32Array(nodeCount);
  const childIndices: number[] = [];

  for (let i = 0; i < nodeCount; i++) {
    nodeChildStart[i] = childIndices.length;
    for (const childId of graph.nodeChildren[i]!) childIndices.push(childId);
  }

  const childIndexArray = Uint32Array.from(childIndices);
  const header = Uint32Array.from([MAGIC, graph.rootId, nodeCount, childIndexArray.length]);

  return Buffer.concat([
    Buffer.from(header.buffer),
    Buffer.from(graph.nodeMask.buffer),
    Buffer.from(nodeChildStart.buffer),
    Buffer.from(childIndexArray.buffer),
  ]);
}

const { root, wordCount } = buildTrie(words);
const graph = minimize(root);
const outBuffer = serialize(graph);

const outPath = resolve("public/wordlist.trie.bin");
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, outBuffer);

const edgeCount = graph.nodeChildren.reduce((sum, children) => sum + children.length, 0);
console.log(`kept ${wordCount.toLocaleString()} words (${MIN_WORD_LEN}-${MAX_WORD_LEN} letters, a-z only)`);
console.log(`dawg: ${graph.nodeMask.length.toLocaleString()} nodes, ${edgeCount.toLocaleString()} edges`);
console.log(`output: ${outBuffer.length.toLocaleString()} bytes -> ${outPath}`);
