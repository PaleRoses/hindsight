import { describe, expect, it } from "vitest";
import { parsePageList, buildKnowledgePreamble, buildRosterRefresh } from "./knowledge-injection";

describe("parsePageList", () => {
  it("extracts {id,title} from the page list shape, tolerating junk", () => {
    const raw = {
      items: [
        { id: "p1", name: "Component map" },
        { id: "p2", name: "Core concepts" },
        { nope: 1 },
      ],
    };
    expect(parsePageList(raw)).toEqual([
      { id: "p1", title: "Component map" },
      { id: "p2", title: "Core concepts" },
    ]);
  });
  it("carries the server's staleness verdict, and only when it gave one", () => {
    const items = [
      { id: "p1", name: "Component map", is_stale: true },
      { id: "p2", name: "Core concepts", is_stale: false },
      { id: "p3", name: "Key decisions" },
      { id: "p4", name: "Conventions", is_stale: "yes" },
    ];
    expect(parsePageList({ items })).toEqual([
      { id: "p1", title: "Component map", stale: true },
      { id: "p2", title: "Core concepts", stale: false },
      { id: "p3", title: "Key decisions" },
      { id: "p4", title: "Conventions" },
    ]);
  });
  it("returns [] for null/garbage", () => {
    expect(parsePageList(null)).toEqual([]);
    expect(parsePageList(42 as unknown)).toEqual([]);
  });
});

describe("buildKnowledgePreamble", () => {
  it("lists the pages and gives a when-to-call guide for the FULL tool suite", () => {
    const out = buildKnowledgePreamble([{ id: "p1", title: "Component map" }]);
    expect(out).toContain("<hindsight_knowledge>");
    expect(out).toContain("Component map");
    expect(out).toContain("p1");
    // Every meaningful tool must be named with a when-to-call, not just pages.
    expect(out).toContain("hindsight_list_knowledge_pages");
    expect(out).toContain("hindsight_read_knowledge_page");
    expect(out).toContain("hindsight_reflect");
    expect(out).toContain("hindsight_capture_initiative");
    expect(out).toContain("hindsight_ingest_document");
  });
  it("tells the agent to recapture an initiative when the plan changes mid-work", () => {
    // Same contract as the MCP tool description (knowledge-tools.ts) — the two must not drift.
    for (const out of [
      buildKnowledgePreamble([{ id: "p1", title: "Component map" }]),
      buildRosterRefresh([]),
    ]) {
      expect(out).not.toMatch(/call this ONCE/i);
      expect(out).toMatch(/call it AGAIN with relates_to_page_id/i);
      expect(out).toMatch(/goal, scope, or rationale materially changes/i);
    }
  });

  it("has an empty-state line when there are no pages", () => {
    const out = buildKnowledgePreamble([]);
    expect(out).toMatch(/no knowledge pages yet|still learning/i);
  });

  it("checks knowledge pages before reflection in tool-only mode", () => {
    for (const out of [
      buildKnowledgePreamble([{ id: "p1", title: "Component map" }], {
        reflectOnNewGoals: true,
      }),
      buildRosterRefresh([{ id: "p1", title: "Component map" }], {
        reflectOnNewGoals: true,
      }),
    ]) {
      expect(out).toMatch(/new task or goal.*knowledge pages FIRST/is);
      expect(out).toMatch(/hindsight_reflect only when.*pages are too shallow/is);
      // No `s` flag ON PURPOSE: this must stay a per-LINE guard against the old wording
      // ("call hindsight_reflect with that goal FIRST"). With `s` it would span newlines and
      // match the legitimate "hindsight_reflect ..." / "FIRST STOP" lines further down the guide.
      expect(out).not.toMatch(/hindsight_reflect.*FIRST/);
    }
  });
});

describe("buildRosterRefresh", () => {
  it("lists current pages and re-states the full tool guide", () => {
    const out = buildRosterRefresh([{ id: "p1", title: "Component map" }]);
    expect(out).toContain("Component map");
    expect(out).toContain("p1");
    for (const tool of [
      "hindsight_list_knowledge_pages",
      "hindsight_read_knowledge_page",
      "hindsight_capture_initiative",
      "hindsight_ingest_document",
    ]) {
      expect(out).toContain(tool);
    }
  });
  it("still emits the full tool guide when there are no pages yet (no roster, but the guide persists)", () => {
    const out = buildRosterRefresh([]);
    expect(out).toContain("<hindsight_knowledge_refresh>");
    expect(out).toContain("hindsight_capture_initiative");
    expect(out).toContain("hindsight_ingest_document");
    // No roster block when there are no pages.
    expect(out).not.toContain("Current Hindsight knowledge pages");
  });
});

describe("staleness in the injected rosters", () => {
  const PAGES = [
    { id: "p1", title: "Component map", stale: true },
    { id: "p2", title: "Core concepts", stale: false },
  ];

  it("marks a stale page in the SessionStart roster and explains the mark once", () => {
    const out = buildKnowledgePreamble(PAGES);
    expect(out).toContain("- Component map (p1) — STALE");
    expect(out).toContain("- Core concepts (p2)");
    expect(out).not.toContain("- Core concepts (p2) — STALE");
    expect(out.match(/Pages marked STALE/g)).toHaveLength(1);
  });

  /** The roster re-appears on every refresh, so the mark must survive there — and a legend
   *  printed unconditionally is boilerplate, which stops being read. */
  it("marks it in the periodic refresh too, and drops the legend when nothing is flagged", () => {
    expect(buildRosterRefresh(PAGES)).toContain("- Component map (p1) — STALE");
    for (const out of [buildKnowledgePreamble([PAGES[1]]), buildRosterRefresh([PAGES[1]])]) {
      expect(out).not.toContain("STALE");
      expect(out).toContain("- Core concepts (p2)");
    }
  });
});
