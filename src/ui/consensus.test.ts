import { describe, expect, it } from "vitest";
import { ScanConsensus, type Reading } from "./consensus";

function reading(letters: string, confidence = 0.9): Reading {
  const cells = letters.split("");
  return {
    letters: cells.map((letter) => (letter === "?" ? "" : letter)),
    confidences: cells.map((letter) => (letter === "?" ? null : confidence)),
  };
}

const options = {
  windowSize: 6,
  minFrames: 3,
  minAgreementRatio: 0.6,
  minMeanConfidence: 0.5,
  minFrameConfidence: 0.4,
};

describe("ScanConsensus", () => {
  it("locks once enough frames agree", () => {
    const consensus = new ScanConsensus(options);
    expect(consensus.add(reading("ABCDE"))).toBeNull();
    expect(consensus.add(reading("ABCDE"))).toBeNull();

    const locked = consensus.add(reading("ABCDE"));
    expect(locked).not.toBeNull();
    expect(locked!.letters.join("")).toBe("ABCDE");
    expect(locked!.frames).toBe(3);
  });

  it("tolerates a single flickering cell instead of deadlocking on it", () => {
    // The regression that made the first scanner never lock: one die reading
    // differently on one frame must not veto 24 settled cells.
    const consensus = new ScanConsensus(options);
    consensus.add(reading("ABCDE"));
    consensus.add(reading("ABCDX"));
    // 2 of 3 votes for E clears the 0.6 agreement bar, so the outvoted frame
    // costs nothing. Under the old whole-board rule this never locked at all.
    const locked = consensus.add(reading("ABCDE"));

    expect(locked).not.toBeNull();
    expect(locked!.letters.join("")).toBe("ABCDE");
  });

  it("refuses to lock while a cell is genuinely split", () => {
    const consensus = new ScanConsensus(options);
    consensus.add(reading("ABCDE"));
    consensus.add(reading("ABCDX"));
    consensus.add(reading("ABCDE"));
    consensus.add(reading("ABCDX"));

    const progress = consensus.progress();
    expect(progress.unsettled).toBe(1);
    expect(progress.cells[4]!.agreement).toBeCloseTo(0.5, 6);
  });

  it("discards a junk frame instead of voting with it", () => {
    // Measured on-device: scans open with 0.18-0.28-confidence reads while the
    // camera focuses. Those must not enter the window at all — they cannot
    // outvote anything, but they delay every cell from settling.
    const consensus = new ScanConsensus(options);
    consensus.add(reading("ABCDE", 0.2));
    consensus.add(reading("XXXXX", 0.15));
    expect(consensus.frames).toBe(0);

    consensus.add(reading("ABCDE"));
    consensus.add(reading("ABCDE"));
    expect(consensus.add(reading("ABCDE"))).not.toBeNull();
  });

  it("refuses to lock on a consistent but low-confidence reading", () => {
    // Consistency alone cannot be the rule: a die misread the same way every
    // frame agrees with itself perfectly.
    // 0.45 clears the per-frame admission bar (0.4) but not the lock bar (0.5).
    const consensus = new ScanConsensus(options);
    for (let i = 0; i < 5; i++) expect(consensus.add(reading("ABCDE", 0.45))).toBeNull();
    expect(consensus.frames).toBeGreaterThan(0);
  });

  it("lets a cell abstain without losing the frame's other votes", () => {
    const consensus = new ScanConsensus(options);
    consensus.add(reading("ABCDE"));
    consensus.add(reading("AB?DE")); // one die occluded
    consensus.add(reading("ABCDE"));

    // C has only two voters, one short of minFrames, so the board is not
    // settled yet — but A, B, D and E all are.
    const progress = consensus.progress();
    expect(progress.unsettled).toBe(1);
    expect(progress.cells[0]!.settled).toBe(true);

    expect(consensus.add(reading("ABCDE"))).not.toBeNull();
  });

  it("re-settles on a new board rather than deadlocking when it changes", () => {
    // Old votes fall out of the window, so swapping the board in front of the
    // camera eventually locks on the new letters.
    const consensus = new ScanConsensus(options);
    for (let i = 0; i < 6; i++) consensus.add(reading("AAAAA"));
    for (let i = 0; i < 6; i++) consensus.add(reading("BBBBB"));

    const locked = consensus.add(reading("BBBBB"));
    expect(locked!.letters.join("")).toBe("BBBBB");
  });

  it("reports mean confidence of the winning letters", () => {
    const consensus = new ScanConsensus({ ...options, minFrames: 2, minMeanConfidence: 0 });
    consensus.add(reading("AB", 0.6));
    const locked = consensus.add(reading("AB", 0.8));
    expect(locked!.meanConfidence).toBeCloseTo(0.7, 6);
  });
});
