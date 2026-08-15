import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Trie } from "./trie";

/** Loads the real compiled dictionary (built by `npm run build:wordlist`) for tests. */
export function loadCompiledTrie(): Trie {
  const bytes = readFileSync(resolve("public/wordlist.trie.bin"));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return Trie.fromBuffer(buffer);
}
