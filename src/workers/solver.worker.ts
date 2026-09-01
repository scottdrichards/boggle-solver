/// <reference lib="webworker" />
import { createBoard } from "../boggle/board";
import { scoreWords, totalScore } from "../boggle/score";
import { solve } from "../boggle/solve";
import { Trie } from "../boggle/trie";
import type { WorkerRequest, WorkerResponse } from "./solverProtocol";

let trie: Trie | null = null;

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      trie = Trie.fromBuffer(message.buffer);
      post({ type: "dictionary-status", status: "ready" });
    } catch (error) {
      console.error("failed to parse dictionary", error);
      post({ type: "dictionary-status", status: "error" });
    }
    return;
  }

  if (message.type === "solve") {
    if (!trie) return; // client gates solve requests on dictionary-status "ready"
    const board = createBoard(message.tiles);
    const found = solve(board, trie, message.minWordLength);
    const scored = scoreWords(found);
    post({
      type: "solve-result",
      requestId: message.requestId,
      words: scored.map((word) => ({ word: word.word, path: [...word.path], points: word.points })),
      totalPoints: totalScore(scored),
    });
  }
});
