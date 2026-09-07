import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MemorySharing } from "./host-client";

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

/**
 * Directed sharing: one owner writes ONE statement into another owner's bank, and nothing else
 * about either identity moves. The capability IS the authorisation — minted from the registry the
 * host bound to — but every send re-checks it against the live file, because the edge may have
 * been revoked, the recipient re-pointed, or either side switched off since.
 */
describe("createSharing", () => {
  const REGISTRY = {
    principals: {
      alpha: { bankId: "alpha-bank", shareTo: ["beta"] },
      beta: { bankId: "beta-bank" },
      gamma: { bankId: "gamma-bank" },
    },
    principal: "alpha",
  };
  const shared = (over: Record<string, unknown> = {}) => ({
    apiUrl: "http://server",
    apiToken: "alpha-key",
    ...REGISTRY,
    ...over,
  });

  type Sent = { url: string; auth: string | null; body: Record<string, unknown> };

  function captureFetch(): Sent[] {
    const sent: Sent[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        sent.push({
          url: String(url),
          auth: new Headers(init.headers).get("Authorization"),
          body: JSON.parse(String(init.body ?? "{}")) as Record<string, unknown>,
        });
        return new Response(JSON.stringify({ operation_id: "op-1", total: 0 }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      })
    );
    return sent;
  }

  /** The capability the alpha -> beta edge authorises, read from whatever the config file now
   *  says — so a case that authorises nothing fails here instead of asserting on nothing. */
  async function bind(): Promise<MemorySharing> {
    const { createSharing, resolveHostMemory } = await loadFactory();
    const sharing = createSharing("dsh", root, resolveHostMemory("dsh", root));
    if (!sharing) throw new Error("expected a sharing capability for alpha -> beta");
    return sharing;
  }

  it("mints no capability without a bound owner holding an edge", async () => {
    for (const cfg of [
      { apiUrl: "http://server" }, // no owners at all: the legacy per-repo route
      // a registered owner with no edge: sharing is opted into per owner, never implied
      {
        apiUrl: "http://server",
        principals: { alpha: { bankId: "alpha-bank" } },
        principal: "alpha",
      },
      shared({ disabled: true }),
    ]) {
      writeConfig(cfg);
      const { createSharing, resolveHostMemory } = await loadFactory();
      expect(createSharing("dsh", root, resolveHostMemory("dsh", root))).toBeUndefined();
    }
  });

  it("writes the statement into the recipient's bank, attributed to both owners", async () => {
    writeConfig(shared({ banks: { "beta-bank": { apiToken: "beta-key" } } }));
    const sharing = await bind();
    expect(sharing.recipients).toEqual(["beta"]);

    const sent = captureFetch();
    const out = await sharing.send({
      recipient: "beta",
      content: "the fixture digests moved",
      context: "cut 4",
    });

    // Exactly one request: a share writes a document and does not seed, configure or probe the
    // recipient's bank — that bank belongs to an identity this host does not act for.
    expect(sent).toHaveLength(1);
    expect(sent[0].url).toBe("http://server/v1/default/banks/beta-bank/memories");
    expect(sent[0].auth).toBe("Bearer beta-key");

    const item = (sent[0].body.items as Record<string, unknown>[])[0];
    expect(out).toEqual({ recipient: "beta", documentId: item.document_id, operationId: "op-1" });
    expect(item.strategy).toBe("document");
    expect(String(item.content)).toContain("the fixture digests moved");
    expect(String(item.context)).toContain("cut 4");
    // The document is the RECIPIENT's and records who put it there: beta's recall must be able to
    // answer "who told me this", and alpha must never read as the owner of beta's document.
    expect([...(item.tags as string[])].sort()).toEqual([
      "harness:dsh",
      "principal:beta",
      "shared-by:alpha",
      "shared-with:beta",
      "source:upload",
    ]);
    expect(item.metadata).toEqual({
      harness: "dsh",
      principal: "beta",
      shared_by: "alpha",
      shared_with: "beta",
    });
    vi.unstubAllGlobals();
  });

  it("writes under the recipient bank's policy, not the sender's", async () => {
    // The statement lands in beta's bank, so beta's `banks.<id>` section governs how it is
    // consolidated there. Alpha's section describes a different bank and must not follow it across.
    writeConfig(
      shared({
        banks: {
          "alpha-bank": { observationScopes: "all_combinations" },
          "beta-bank": { observationScopes: "per_tag" },
        },
      })
    );
    const sharing = await bind();
    const sent = captureFetch();
    await sharing.send({ recipient: "beta", content: "c" });
    const item = (sent[0].body.items as Record<string, unknown>[])[0];
    expect(item.observation_scopes).toBe("per_tag");
    vi.unstubAllGlobals();
  });

  it("leaves the sender's own route exactly where it was", async () => {
    writeConfig(shared({ banks: { "beta-bank": { apiToken: "beta-key" } } }));
    const { createSharing, resolveHostMemory } = await loadFactory();
    const memory = resolveHostMemory("dsh", root);
    const sent = captureFetch();
    await createSharing("dsh", root, memory)?.send({ recipient: "beta", content: "c" });
    expect(sent.map((s) => s.url)).toEqual(["http://server/v1/default/banks/beta-bank/memories"]);

    // Same host, immediately afterwards: still its own bank, still its own credential.
    sent.length = 0;
    await memory.client.activeOperations();
    expect(sent).toEqual([
      {
        url: expect.stringMatching(
          "^http://server/v1/default/banks/alpha-bank/operations(?:\\?|$)"
        ),
        auth: "Bearer alpha-key",
        body: {},
      },
    ]);
    vi.unstubAllGlobals();
  });

  it("refuses a recipient the live registry no longer permits, without writing", async () => {
    writeConfig(shared());
    const sharing = await bind();
    const sent = captureFetch();

    // Never advertised: the capability's own list is the authorisation, so a registered owner the
    // edge does not name stays unreachable through it.
    await expect(sharing.send({ recipient: "gamma", content: "c" })).rejects.toThrow();

    // The edge is revoked mid-session.
    writeConfig(
      shared({ principals: { alpha: { bankId: "alpha-bank" }, beta: { bankId: "beta-bank" } } })
    );
    await expect(sharing.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    // Edge intact, but the recipient's bank moved: the statement would land in a bank this host
    // was never authorised to write to, under an identity that never agreed to hold it.
    writeConfig(
      shared({
        principals: {
          alpha: { bankId: "alpha-bank", shareTo: ["beta"] },
          beta: { bankId: "beta-elsewhere" },
        },
      })
    );
    await expect(sharing.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    expect(sent).toEqual([]);
    vi.unstubAllGlobals();
  });

  it("refuses once the sender's own binding moved under it, without writing", async () => {
    writeConfig(shared());
    const sharing = await bind();
    const sent = captureFetch();

    // The sender is re-pointed: a capability minted for alpha-bank's owner no longer speaks for it.
    writeConfig(
      shared({
        principals: {
          alpha: { bankId: "moved", shareTo: ["beta"] },
          beta: { bankId: "beta-bank" },
        },
      })
    );
    await expect(sharing.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    writeConfig(shared({ disabled: true }));
    await expect(sharing.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    expect(sent).toEqual([]);
    vi.unstubAllGlobals();
  });

  it("refuses a recipient that is switched off or lives on another server, without writing", async () => {
    // `banks.<recipient bank>` is the recipient's own opt-out, and a share must honour it exactly
    // as the recipient's own host would — and never carry their memory to a second endpoint.
    const sent = captureFetch();
    writeConfig(shared({ banks: { "beta-bank": { disabled: true } } }));
    const off = await bind();
    await expect(off.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    writeConfig(shared({ banks: { "beta-bank": { apiUrl: "http://elsewhere" } } }));
    const away = await bind();
    await expect(away.send({ recipient: "beta", content: "c" })).rejects.toThrow();

    expect(sent).toEqual([]);
    vi.unstubAllGlobals();
  });

  it("re-resolves the RECIPIENT's credential on a 401, never the sender's", async () => {
    writeConfig(shared({ banks: { "beta-bank": { apiToken: "beta-key" } } }));
    const sharing = await bind();

    // The recipient's key rotates under the in-flight write; only the new one is accepted. Signing
    // the retry with alpha-key would write beta's document with the sender's credential.
    const auths: (string | null)[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const auth = new Headers(init.headers).get("Authorization");
        auths.push(auth);
        if (auth !== "Bearer beta-rotated") {
          writeConfig(shared({ banks: { "beta-bank": { apiToken: "beta-rotated" } } }));
          return new Response("{}", {
            status: 401,
            headers: { "Content-Type": "application/json" },
          });
        }
        return new Response(JSON.stringify({ operation_id: "op-2" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      })
    );

    const out = await sharing.send({ recipient: "beta", content: "c" });
    expect(out.operationId).toBe("op-2");
    expect(auths).toEqual(["Bearer beta-key", "Bearer beta-rotated"]);
    vi.unstubAllGlobals();
  });
});
