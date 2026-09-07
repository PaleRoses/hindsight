/**
 * Wait out the server-side follow-on work (consolidation, page refreshes) a run's own drain does
 * not cover. A bank-wide ZERO is the wrong target: retries parked with a far-future `next_retry_at`
 * and other sessions' work floor the count, so waiting for zero burns the whole deadline while the
 * caller holds the per-bank lock, locking every session started in that window out of ingesting.
 * Wait for PROGRESS: once the backlog stops shrinking, the rest is somebody else's.
 */
export type SettleOutcome = "drained" | "stalled" | "timeout";

export type SettleResult = { outcome: SettleOutcome; active: number; polls: number };

export interface SettleOptions {
  pollMs?: number;
  /** Consecutive polls without a NEW LOW-WATER mark before the backlog counts as stalled. */
  stallPolls?: number;
  maxMs?: number;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  log?: (message: string) => void;
}

export const SETTLE_POLL_MS = 5_000;
export const SETTLE_STALL_POLLS = 6; // 30s without progress
export const SETTLE_MAX_MS = 15 * 60 * 1000;

export async function settleForProgress(
  activeOperations: () => Promise<number>,
  opts: SettleOptions = {}
): Promise<SettleResult> {
  const { pollMs = SETTLE_POLL_MS, stallPolls = SETTLE_STALL_POLLS, maxMs = SETTLE_MAX_MS } = opts;
  const { now = Date.now, log = () => {} } = opts;
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const stallWindow = (stallPolls * pollMs) / 1000;
  const deadline = now() + maxMs;
  let lowWater = Infinity;
  let stalled = 0;
  let polls = 0;

  for (;;) {
    const active = await activeOperations();
    polls++;
    if (active === 0) return { outcome: "drained", active: 0, polls };

    // Low-water mark, not the previous reading: the count RISES as new work arrives, and a dip
    // below the last reading that is still above the floor is not progress on this run's backlog.
    if (active < lowWater) {
      lowWater = active;
      stalled = 0;
    } else if (++stalled >= stallPolls) {
      log(`${active} op(s) active, none cleared in ${stallWindow}s — not this run's, proceeding`);
      return { outcome: "stalled", active, polls };
    }

    if (now() >= deadline) {
      log(`${active} server-side op(s) still active at settle timeout — proceeding`);
      return { outcome: "timeout", active, polls };
    }
    log(`waiting for ${active} server-side op(s) to settle …`);
    await sleep(pollMs);
  }
}
