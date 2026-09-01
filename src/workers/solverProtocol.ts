export interface InitMessage {
  type: "init";
  buffer: ArrayBuffer;
}

export interface SolveRequest {
  type: "solve";
  requestId: number;
  tiles: string[];
  minWordLength: number;
}

export type WorkerRequest = InitMessage | SolveRequest;

export interface DictionaryStatusMessage {
  type: "dictionary-status";
  status: "ready" | "error";
}

export interface SolveResultWord {
  word: string;
  path: number[];
  points: number;
}

export interface SolveResultMessage {
  type: "solve-result";
  requestId: number;
  words: SolveResultWord[];
  totalPoints: number;
}

export type WorkerResponse = DictionaryStatusMessage | SolveResultMessage;
