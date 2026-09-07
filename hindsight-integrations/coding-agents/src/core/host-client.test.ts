import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let root: string;
let cfgPath: string;

/** `CONFIG_PATH` is resolved from the environment when core/config is first imported, so each case
 *  points HINDSIGHT_CONFIG at its own file and re-imports the module graph. */
async function loadFactory() {
  vi.resetModules();
  process.env.HINDSIGHT_CONFIG = cfgPath;
  return import("./host-client");
}

function writeConfig(value: unknown): void {
  mkdirSync(join(cfgPath, ".."), { recursive: true });
  writeFileSync(cfgPath, JSON.stringify(value));
}

const ENV = { ...process.env };

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-host-"));
  cfgPath = join(root, "coding-agent.json");
});

afterEach(() => {
  process.env = { ...ENV };
  rmSync(root, { recursive: true, force: true });
});

describe("resolveHostMemory", () => {
  it("binds the client to the selected owner's bank, whatever workspace it is serving", async () => {
    // The owner's bank is the identity's, so the request path must carry it verbatim — the repo
    // this host happens to be opened on has no say, and the id needs no shape of its own.
    writeConfig({
      apiUrl: "http://server",
      apiToken: "k",
      principals: { alpha: { bankId: "Alpha::Personal Memory" } },
      principal: "alpha",
    });
    const { resolveHostMemory } = await loadFactory();

    const { bankId, client } = resolveHostMemory("dsh", root);
    expect(bankId).toBe("Alpha::Personal Memory");

    const requests: { url: string; auth: string | null }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        requests.push({ url: String(url), auth: new Headers(init.headers).get("Authorization") });
        return new Response(JSON.stringify({ total: 0 }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      })
    );
    await client.activeOperations();
    expect(requests).toEqual([
      {
        url: expect.stringMatching(
          "^http://server/v1/default/banks/Alpha%3A%3APersonal%20Memory/operations(?:\\?|$)"
        ),
        auth: "Bearer k",
      },
    ]);
    vi.unstubAllGlobals();
  });

  it("stays inert when the selector names no owner, rather than serving the repo's bank", async () => {
    writeConfig({
      apiUrl: "http://server",
      principals: { alpha: { bankId: "Alpha::Personal" } },
      principal: "ghost",
    });
    const { resolveHostMemory } = await loadFactory();

    const { cfg, bankId } = resolveHostMemory("dsh", root);
    expect(cfg.disabled).toBe(true);
    expect(bankId).toBe("");
  });

  it("keeps optInOnly closed for an owner-routed host too", async () => {
    // A principal names a bank, not an approved project: the privacy switch still fails closed.
    writeConfig({
      optInOnly: true,
      optInPaths: ["/somewhere/else"],
      principals: { alpha: { bankId: "Alpha::Personal" } },
      principal: "alpha",
    });
    const { resolveHostMemory } = await loadFactory();

    expect(resolveHostMemory("dsh", root).cfg.disabled).toBe(true);
  });

  it("enforces optInOnly for every host, not just the ones that remembered to pass a directory", async () => {
    // dsh called applyBankConfig WITHOUT the directory, so `optInOnly` was never enforced there:
    // an unapproved repo still got a bank and still had memories written for it.
    writeConfig({ optInOnly: true, optInPaths: ["/somewhere/else"] });
    const { resolveHostMemory } = await loadFactory();

    expect(resolveHostMemory("dsh", root).cfg.disabled).toBe(true);
  });

  it("does not derive a bank when memory is disabled — that path is a zero-overhead baseline", async () => {
    // Bank derivation shells out to git. `disabled` promises the same agent with NO memory work,
    // which is what makes it usable as an A/B baseline, so it must stop before that.
    writeConfig({ disabled: true });
    const { resolveHostMemory } = await loadFactory();

    const { cfg, bankId } = resolveHostMemory("dsh", root);
    expect(cfg.disabled).toBe(true);
    expect(bankId).toBe("");
  });

  it("re-resolves the token from the live config, so a rotation does not need a restart", async () => {
    writeConfig({ apiUrl: "http://server", apiToken: "old-key" });
    const { resolveHostMemory } = await loadFactory();
    const { client } = resolveHostMemory("dsh", root);
    expect(client.apiToken).toBe("old-key");

    // The operator rotates the credential while the host keeps running; the next 401 picks it up.
    writeConfig({ apiUrl: "http://server", apiToken: "new-key" });
    const calls: (string | null)[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const auth = new Headers(init.headers).get("Authorization");
        calls.push(auth);
        return new Response(JSON.stringify(auth === "Bearer new-key" ? { ok: true } : {}), {
          status: auth === "Bearer new-key" ? 200 : 401,
          headers: { "Content-Type": "application/json" },
        });
      })
    );

    await client.req("GET", "http://server/thing");
    expect(calls).toEqual(["Bearer old-key", "Bearer new-key"]);
    vi.unstubAllGlobals();
  });

  it("honours a per-bank apiToken on re-resolution, not just on the first read", async () => {
    // `banks.<id>.apiToken` is a legitimate override (it is not stripped by BANK_OVERRIDE_EXCLUDED),
    // so a provider that re-read only the top level would hand back the wrong credential.
    const { resolveHostMemory: probe } = await loadFactory();
    writeConfig({ apiUrl: "http://server", apiToken: "global" });
    const bankId = probe("dsh", root).bankId;

    writeConfig({
      apiUrl: "http://server",
      apiToken: "global",
      banks: { [bankId]: { apiToken: "per-bank" } },
    });
    const { resolveHostMemory } = await loadFactory();
    expect(resolveHostMemory("dsh", root).client.apiToken).toBe("per-bank");
  });

  it("keeps an owner-routed host on the binding it was built with, credential rotation aside", async () => {
    // A long-lived host holds a SNAPSHOT: it goes on serving the owner and bank it resolved at
    // startup, and nothing re-reads the registry per request. What may still change under it is
    // the credential — what may not is which identity's memory it signs for.
    const bound = { principals: { alpha: { bankId: "alpha-bank" } }, principal: "alpha" };
    writeConfig({ apiUrl: "http://server", apiToken: "old-key", ...bound });
    const { resolveHostMemory } = await loadFactory();
    const { client } = resolveHostMemory("dsh", root);

    // The server accepts exactly one credential at a time, so a 200 proves which one was sent.
    let accepted = "rotated-key";
    const calls: { url: string; auth: string | null }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        const auth = new Headers(init.headers).get("Authorization");
        calls.push({ url: String(url), auth });
        return new Response(JSON.stringify({ total: 0 }), {
          status: auth === `Bearer ${accepted}` ? 200 : 401,
          headers: { "Content-Type": "application/json" },
        });
      })
    );

    writeConfig({ apiUrl: "http://server", apiToken: "rotated-key", ...bound });
    await client.activeOperations();
    expect(calls.map((c) => c.auth)).toEqual(["Bearer old-key", "Bearer rotated-key"]);

    // The owner is re-pointed at another bank, with its own credential, mid-session.
    writeConfig({
      apiUrl: "http://server",
      apiToken: "other-key",
      principals: { alpha: { bankId: "other-bank" } },
      principal: "alpha",
    });
    calls.length = 0;
    await client.activeOperations();
    // Still serving the bank it bound to — a re-pointed registry moves nothing under a live host.
    expect(calls).toEqual([
      {
        url: expect.stringMatching(
          "^http://server/v1/default/banks/alpha-bank/operations(?:\\?|$)"
        ),
        auth: "Bearer rotated-key",
      },
    ]);

    // Forced to re-resolve by a 401, it still refuses to sign with the re-pointed owner's key:
    // that request would read and write another identity's memory.
    accepted = "other-key";
    calls.length = 0;
    await client.activeOperations().catch(() => {});
    expect(calls.map((c) => c.auth)).toEqual(["Bearer rotated-key"]);
    vi.unstubAllGlobals();
  });
});
