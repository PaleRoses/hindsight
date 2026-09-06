import { describe, expect, it } from "vitest";
import { settleForProgress } from "./settle";

/** Deterministic clock + sleep: the wait is a timing policy, so it is tested by advancing a fake
 *  clock rather than by actually waiting. `sleep` advances the same clock `now` reads. */
function harness(readings: number[]) {
  let clock = 0;
  const logs: string[] = [];
  let i = 0;
  return {
    logs,
    elapsed: () => clock,
    calls: () => i,
    opts: {
      pollMs: 5_000,
      stallPolls: 6,
      maxMs: 15 * 60 * 1000,
      now: () => clock,
      sleep: async (ms: number) => {
        clock += ms;
      },
      log: (m: string) => logs.push(m),
    },
    // Readings run out by repeating the last one — a bank that simply stays where it is.
    active: async () => readings[Math.min(i++, readings.length - 1)],
  };
}

describe("settleForProgress", () => {
  it("returns immediately when the bank is already clear", async () => {
    const h = harness([0]);
    const r = await settleForProgress(h.active, h.opts);
    expect(r).toEqual({ outcome: "drained", active: 0, polls: 1 });
    expect(h.elapsed()).toBe(0);
  });

  it("keeps waiting while the backlog is shrinking, then reports it drained", async () => {
    const h = harness([5, 4, 3, 2, 1, 0]);
    const r = await settleForProgress(h.active, h.opts);
    expect(r.outcome).toBe("drained");
    expect(r.polls).toBe(6);
  });

  /** The defect this exists to prevent: a bank whose count never reaches zero because ~370 retains
   *  are parked with next_retry_at in 2027. Waiting for zero burned the full 15-minute deadline on
   *  EVERY session start and held the per-bank lock for all of it. */
  it("stops after the stall window when a parked backlog never clears", async () => {
    const h = harness([373]);
    const r = await settleForProgress(h.active, h.opts);
    expect(r.outcome).toBe("stalled");
    expect(r.active).toBe(373);
    // 6 stall polls at 5s: seconds, not the 15 minutes the zero-target version spent.
    expect(h.elapsed()).toBe(30_000);
    expect(h.elapsed()).toBeLessThan(h.opts.maxMs);
    expect(h.logs.at(-1)).toMatch(/none cleared in 30s/);
  });

  /** A rise is new work arriving, not progress on the backlog being waited out — it must not reset
   *  the stall counter, or a busy bank keeps the loop alive indefinitely. */
  it("does not treat a rising count as progress", async () => {
    const h = harness([300, 301, 302, 303, 304, 305, 306, 307]);
    const r = await settleForProgress(h.active, h.opts);
    expect(r.outcome).toBe("stalled");
    expect(h.elapsed()).toBe(30_000);
  });

  it("resets the stall counter on every new low-water mark", async () => {
    // Five flat polls, one drop, five flat again: never six consecutive without progress.
    const h = harness([100, 100, 100, 100, 100, 99, 99, 99, 99, 99, 0]);
    const r = await settleForProgress(h.active, h.opts);
    expect(r.outcome).toBe("drained");
  });

  /** Pins WHY the stall detector exists. Disabling it reproduces the original wait-for-zero policy
   *  exactly: against the same parked backlog it spends the entire 900s deadline — which is what
   *  the live logs showed ("deepen complete in 901.4s") on every single session start. */
  it("without the stall detector, the same backlog burns the whole deadline", async () => {
    const h = harness([373]);
    const r = await settleForProgress(h.active, { ...h.opts, stallPolls: Infinity });
    expect(r.outcome).toBe("timeout");
    expect(h.elapsed()).toBe(900_000); // 30x the 30_000 the shipped policy spends
  });

  it("honours the hard deadline while progress is still being made", async () => {
    // Strictly decreasing forever: never stalls, so only the ceiling can stop it.
    let n = 1_000_000;
    const h = harness([]);
    const r = await settleForProgress(async () => n--, h.opts);
    expect(r.outcome).toBe("timeout");
    expect(h.elapsed()).toBeGreaterThanOrEqual(h.opts.maxMs);
    expect(h.logs.at(-1)).toMatch(/still active at settle timeout/);
  });
});
