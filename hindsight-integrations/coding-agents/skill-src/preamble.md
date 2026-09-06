---
name: hindsight-coding-agent
description: How this machine's Hindsight coding-agent memory works — the plugin behind the 🧠 banner. Use when the user says "store/remember this in hindsight", asks what the memory/knowledge pages are, wants to configure per-repo memory (disable, rename banks, git depth), or something memory-related looks broken.
---

# Hindsight Coding-Agent Memory

This machine runs the `hindsight-coding-agents` plugin: long-term project memory for coding
sessions, backed by a Hindsight server. You (the agent) are already wired into it — this skill
explains what happens automatically, which tools you have, and how to configure or debug it.

## What happens automatically (no action needed)

- **Per-repo memory bank**: each repository resolves to a bank (shown in the session banner:
  `↳ memory bank “coding-agent::<repo>”`). Worktrees share the main repo's bank.
- **Ingestion builds itself**: on first open, the bank is seeded from recent commit messages and a
  read-only codebase survey; every session start, a background engine tops it up (new commits, new
  conversations) and keeps 5 knowledge pages current. There is NO ingest command to run.
- **Session synthesis**: by default, the first prompt of a session triggers one deep memory
  synthesis (`reflect`) injected into context. With `autoReflect=false`, the agent searches the
  knowledge pages first and reflects only when they are too shallow.
- **Write-back**: the session transcript is retained into the bank automatically at session end
  (per-turn on opencode). The user never needs to "save" a conversation.

## Storing things deliberately

When the user says "store this in hindsight" / "remember this":

- The **current conversation** is captured automatically at session end — say so; no tool needed.
- An **external document, notes, or durable findings** → `hindsight_ingest_document(title, content)`.
- A **new feature/initiative being started** → `hindsight_capture_initiative(title, summary)`,
  right after the plan is agreed and before code is written.
- A **plan that materially changed** (goal, scope, or rationale — including mid-implementation) →
  call `hindsight_capture_initiative` again with `relates_to_page_id` set to that initiative's page
  id, summarising the _current_ intent. Same page, updated plan — never a second page. Trivial
  course-corrections don't count.

## Retrieving

- `hindsight_search_knowledge_pages(query)` — FIRST STOP for project questions (components,
  conventions, past decisions, initiatives). Server-side hybrid search, fast.
- `hindsight_read_knowledge_page(page_id)` / `hindsight_list_knowledge_pages` — read pages fully.
- `hindsight_reflect(query)` — deep reasoning over the whole memory for WHY questions and exact
  decided values; slower (seconds), use deliberately.
- Credit visibly whenever memory informs an answer: start that part with
  `🧠 From Hindsight memory (<page>): …` — and never credit memory that didn't contribute.
- A page marked `STALE` in the injected roster, or carrying `"is_stale": true` when you read it,
  has had in-scope memories written since it was last rebuilt: the server already knows it is
  behind the repository. Pages are reliable for _what exists_ — symbols, signatures, structure —
  and much weaker on _how many_, because counts and inventories are the part a synthesis has to
  infer. Verify any specific number against the source before you rely on it.

## Correcting wrong or stale memory

If you verify that something Hindsight served is wrong or outdated (the code, git, or an external
source contradicts it), FIX THE RECORD — don't just ignore it. Call
`hindsight_ingest_document` with:

- **title**: `Correction: <topic>` (e.g. `Correction: retry policy 4xx set`)
- **content**: (1) what memory claimed, (2) what is verifiably true now, (3) the evidence you
  checked (file/commit/output). Quote exact values verbatim.

Be clear about what that does. It records the correction, so a later rebuild of the page is
synthesized over it. It does **not** rewrite the page, and it does not retire the claim it refutes:
retrieval has no supersession mechanism, and the recency term that favours a newer fact is worth a
fraction of a percent against the relevance score. The stale memory stays retrievable until the
page carrying it is next rebuilt.

So do both halves. Ingest the correction **and** state it in your own answer — the next session
will very likely read the page before it reads your document. Do this whenever you catch a wrong
injected memory, a stale knowledge-page claim, or an outdated decision: silent disregard leaves the
trap armed for the next session, and so does assuming the ingest disarmed it.
