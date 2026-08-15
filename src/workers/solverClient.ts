import type { SolveResultWord, WorkerRequest, WorkerResponse } from "./solverProtocol";

export interface SolveOutcome {
  words: SolveResultWord[];
  totalPoints: number;
}

/**
 * Main-thread handle to the solver Web Worker. The worker parses the
 * dictionary and runs solve+score off the main thread; this class just does
 * request/response bookkeeping (a request id per in-flight `solve` call)
 * over postMessage. This is the pattern the CV/ML pipeline worker will
 * reuse later, kept deliberately small rather than a generic RPC layer.
 */
export class SolverClient {
  private readonly worker: Worker;
  private nextRequestId = 0;
  private readonly pending = new Map<number, (outcome: SolveOutcome) => void>();
  private resolveDictionaryReady!: (ready: boolean) => void;
  readonly dictionaryReady: Promise<boolean>;

  constructor() {
    this.worker = new Worker(new URL("./solver.worker.ts", import.meta.url), { type: "module" });
    this.dictionaryReady = new Promise((resolve) => {
      this.resolveDictionaryReady = resolve;
    });

    this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === "dictionary-status") {
        this.resolveDictionaryReady(message.status === "ready");
      } else if (message.type === "solve-result") {
        const resolve = this.pending.get(message.requestId);
        if (resolve) {
          this.pending.delete(message.requestId);
          resolve({ words: message.words, totalPoints: message.totalPoints });
        }
      }
    });
  }

  /** Fetches the compiled dictionary and hands it to the worker (zero-copy transfer). */
  async init(dictionaryUrl: string): Promise<void> {
    const buffer = await (await fetch(dictionaryUrl)).arrayBuffer();
    const message: WorkerRequest = { type: "init", buffer };
    this.worker.postMessage(message, [buffer]);
  }

  solve(tiles: string[]): Promise<SolveOutcome> {
    const requestId = this.nextRequestId++;
    return new Promise((resolve) => {
      this.pending.set(requestId, resolve);
      const message: WorkerRequest = { type: "solve", requestId, tiles };
      this.worker.postMessage(message);
    });
  }
}
