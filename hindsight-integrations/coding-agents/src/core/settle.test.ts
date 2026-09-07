import { describe, expect, it } from "vitest";
import { SETTLE_MAX_MS, settleForProgress } from "./settle";

/** The wait is a timing policy, so it runs against a fake clock that `sleep` advances and `now`
 *  reads, reporting elapsed time and the last log line with the result. An array of readings
 *  cycles — a one-element array is a bank that stays where it is. The poll interval, stall window
 *  and ceiling are left at the exported defaults, which are themselves part of the contract. */
async function settle(readings: number[] | (() => number)) {
  let clock = 0;
  let i = 0;
  const logs: string[] = [];
  const next = typeof readings === "function" ? readings : () => readings[i++ % readings.length];
  const result = await settleForProgress(async () => next(), {
    now: () => clock,
    sleep: async (ms: number) => void (clock += ms),
    log: (m) => logs.push(m),
  });
  return { ...result, elapsed: clock, lastLog: logs.at(-1) };
}

describe("settleForProgress", () => {
  it("drains at zero, and does not sleep when the bank is already clear", async () => {
    expect(await settle([0])).toEqual({ outcome: "drained", active: 0, polls: 1, elapsed: 0 });
    expect(await settle([5, 4, 3, 2, 1, 0])).toMatchObject({ outcome: "drained", polls: 6 });
  });

  /** The defect this exists to prevent: ~370 retains parked with next_retry_at in 2027 keep the
   *  count off zero forever, so waiting for zero burned the whole 15-minute deadline — holding the
   *  per-bank lock for all of it — on every single session start. */
  it("stalls inside the stall window when a parked backlog never clears", async () => {
    const r = await settle([373]);
    expect(r).toMatchObject({ outcome: "stalled", active: 373, polls: 7, elapsed: 30_000 });
    expect(r.lastLog).toMatch(/none cleared in 30s/);
  });

  it("counts only a NEW LOW-WATER mark as progress, not a dip below the last reading", async () => {
    // Oscillating above a floor: every dip beats the poll before it, none beats the low-water
    // mark. Measured against the previous reading this reads as progress and burns the deadline.
    expect(await settle([375, 373, 375])).toMatchObject({ outcome: "stalled", elapsed: 35_000 });
    // A genuine new low does reset the stall counter: five flat, a drop, five flat, then zero.
    const dipping = [100, 100, 100, 100, 100, 99, 99, 99, 99, 99, 0];
    expect(await settle(dipping)).toMatchObject({ outcome: "drained" });
  });

  it("times out at the hard deadline while progress is still being made", async () => {
    let n = 1_000_000; // strictly decreasing forever: never stalls, so only the ceiling stops it
    const r = await settle(() => n--);
    expect(r).toMatchObject({ outcome: "timeout", elapsed: SETTLE_MAX_MS });
    expect(r.lastLog).toMatch(/still active at settle timeout/);
  });
});
