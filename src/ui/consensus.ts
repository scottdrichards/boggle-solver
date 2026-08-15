/**
 * Deciding when a live scan is finished.
 *
 * A scanner that fires on the first confident-looking frame is worse than the
 * four-tap photo flow, because it can commit to a wrong board with no moment
 * at which anyone looked. So the stopping rule is agreement over time: each
 * frame is an independent look with its own noise, blur and lighting, and
 * repeated reads vote out the cells a single frame gets wrong.
 *
 * **Vote per cell, not per board.** The first version of this required all 25
 * letters to be *identical* across N consecutive frames, and in practice it
 * never locked: with 25 cells, one die flickering between two readings is the
 * normal case, and whole-board identity lets that single cell veto every
 * other cell's perfectly stable agreement. Now each cell settles on its own
 * majority and the board locks when every cell has settled — which tolerates
 * a bad frame per cell, and is the same per-cell voting that cleaned up the
 * training labels.
 *
 * Kept free of DOM and tfjs so the rule can be tested directly.
 */

export interface Reading {
  /** One entry per cell, row-major. Empty string means "no confident letter". */
  readonly letters: readonly string[];
  /** Classifier confidence per cell, null where there is no prediction. */
  readonly confidences: readonly (number | null)[];
}

export interface ConsensusOptions {
  /** Readings kept for voting. Older ones fall out, so a board that changes
   * eventually re-settles on the new letters instead of deadlocking. */
  readonly windowSize: number;
  /** Readings required before a lock is possible at all. */
  readonly minFrames: number;
  /** Share of the window's votes a cell's top letter needs to count as
   * settled. Below 1.0 so one bad frame per cell is tolerated. */
  readonly minAgreementRatio: number;
  /** Mean confidence of the winning letters, below which a lock is refused
   * however consistent the votes are. Consistency alone is not enough: a
   * systematically misread die is misread the same way every frame. */
  readonly minMeanConfidence: number;
  /** A frame whose own mean confidence is below this is discarded rather than
   * voted with.
   *
   * Measured on-device: scans routinely opened with frames at 0.18-0.28 mean
   * confidence — the camera still focusing, or the board mid-move — and those
   * junk reads sat in the window outvoting nothing but delaying everything,
   * because a cell cannot settle until they age out. Locks were taking 8-10
   * successful frames against a `minFrames` of 3. A bad read is not weak
   * evidence, it is not evidence. */
  readonly minFrameConfidence: number;
}

export interface CellVote {
  readonly letter: string;
  /** Share of this cell's votes that went to `letter`, 0-1. */
  readonly agreement: number;
  /** Mean confidence across the frames that voted for `letter`. */
  readonly confidence: number | null;
  readonly settled: boolean;
}

export interface ConsensusProgress {
  readonly frames: number;
  readonly cells: CellVote[];
  /** Cells not yet settled — what the scanner is still waiting on. */
  readonly unsettled: number;
  readonly meanConfidence: number;
}

export interface LockedReading {
  readonly letters: string[];
  readonly confidences: (number | null)[];
  readonly frames: number;
  readonly meanConfidence: number;
}

export class ScanConsensus {
  private window: Reading[] = [];

  constructor(private readonly options: ConsensusOptions) {}

  get frames(): number {
    return this.window.length;
  }

  reset(): void {
    this.window = [];
  }

  /** A single frame's own mean confidence, over the cells it actually read. */
  private frameConfidence(reading: Reading): number {
    const known = reading.confidences.filter((value): value is number => value !== null);
    return known.length === 0 ? 0 : known.reduce((sum, value) => sum + value, 0) / known.length;
  }

  /** Per-cell vote state, for the status line and for diagnosing a scan that
   * will not settle — which cell is flickering is the whole question. */
  progress(): ConsensusProgress {
    const cellCount = this.window[0]?.letters.length ?? 0;
    const cells: CellVote[] = [];

    for (let cell = 0; cell < cellCount; cell++) {
      const tally = new Map<string, { count: number; confidence: number }>();
      let voters = 0;
      for (const frame of this.window) {
        const letter = frame.letters[cell];
        if (!letter) continue;
        voters++;
        const entry = tally.get(letter) ?? { count: 0, confidence: 0 };
        entry.count++;
        entry.confidence += frame.confidences[cell] ?? 0;
        tally.set(letter, entry);
      }

      let best = { letter: "", count: 0, confidence: 0 };
      for (const [letter, entry] of tally) {
        if (entry.count > best.count) best = { letter, count: entry.count, confidence: entry.confidence };
      }

      const agreement = voters === 0 ? 0 : best.count / voters;
      cells.push({
        letter: best.letter,
        agreement,
        confidence: best.count === 0 ? null : best.confidence / best.count,
        settled:
          best.letter !== "" &&
          voters >= this.options.minFrames &&
          agreement >= this.options.minAgreementRatio,
      });
    }

    const known = cells.map((c) => c.confidence).filter((v): v is number => v !== null);
    return {
      frames: this.window.length,
      cells,
      unsettled: cells.filter((c) => !c.settled).length,
      meanConfidence: known.length === 0 ? 0 : known.reduce((sum, v) => sum + v, 0) / known.length,
    };
  }

  /** Folds in one frame's reading. Returns the locked board once every cell
   * has settled and the winning letters are confident enough. */
  add(reading: Reading): LockedReading | null {
    if (this.frameConfidence(reading) < this.options.minFrameConfidence) return null;

    // A partly-read frame still carries good votes for the cells it did read,
    // so it is kept rather than dropped; only cells with no letter abstain.
    this.window.push(reading);
    if (this.window.length > this.options.windowSize) this.window.shift();
    if (this.window.length < this.options.minFrames) return null;

    const progress = this.progress();
    if (progress.cells.length === 0 || progress.unsettled > 0) return null;
    if (progress.meanConfidence < this.options.minMeanConfidence) return null;

    return {
      letters: progress.cells.map((cell) => cell.letter),
      confidences: progress.cells.map((cell) => cell.confidence),
      frames: this.window.length,
      meanConfidence: progress.meanConfidence,
    };
  }
}
