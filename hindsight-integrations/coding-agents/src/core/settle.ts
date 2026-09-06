/**
 * Waiting out server-side follow-on work (consolidation, page refreshes) after a run's own
 * operations have drained.
 *
 * The naive version of this waits for the bank's active-operation count to reach ZERO. On a real
 * bank it never does: a retry backlog can be deliberately parked with `next_retry_at` far in the
 * future, and concurrent sessions enqueue work of their own. The count therefore has a floor this
 * run cannot lower, the wait burns its entire deadline every time, and — because the caller holds
 * a per-bank lock while it waits — every session starting inside that window is locked out of
 * ingesting anything at all.
 *
 * So wait for PROGRESS instead. Keep polling while the backlog is still shrinking; once it has
 * stopped shrinking, what remains is somebody else's and no amount of waiting will clear it.
 */

/** Why the wait ended. Reported so a caller can log the difference rather than guess at it. */
export type SettleOutcome =
  /** The bank reached zero active operations — everything, everywhere, drained. */
  | "drained"
  /** The count stopped falling: what is left is not this run's work. The common outcome. */
  | "stalled"
  /** Still shrinking when the hard deadline expired. */
  | "timeout";

export interface SettleResult {
  outcome: SettleOutcome;
  /** Active count at the moment the wait ended (0 when drained). */
  active: number;
  /** Polls performed. Zero means the bank was already clear. */
  polls: number;
}

export interface SettleOptions {
  /** Delay between polls. */
  pollMs?: number;
  /** Consecutive polls without a new low-water mark before declaring the backlog stalled. */
  stallPolls?: number;
  /** Hard ceiling on the whole wait, whatever progress is being made. */
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
  const pollMs = opts.pollMs ?? SETTLE_POLL_MS;
  const stallPolls = opts.stallPolls ?? SETTLE_STALL_POLLS;
  const maxMs = opts.maxMs ?? SETTLE_MAX_MS;
  const now = opts.now ?? (() => Date.now());
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const log = opts.log ?? (() => {});

  const deadline = now() + maxMs;
  let lowWater = Infinity;
  let stalled = 0;
  let polls = 0;

  for (;;) {
    const active = await activeOperations();
    polls++;
    if (active === 0) return { outcome: "drained", active: 0, polls };

    // Low-water mark, not the previous reading: the count RISES as new work arrives, and a rise
    // is not progress on the backlog being waited out.
    if (active < lowWater) {
      lowWater = active;
      stalled = 0;
    } else if (++stalled >= stallPolls) {
      log(
        `${active} op(s) active, none cleared in ${(stallPolls * pollMs) / 1000}s — not this run's, proceeding`
      );
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
