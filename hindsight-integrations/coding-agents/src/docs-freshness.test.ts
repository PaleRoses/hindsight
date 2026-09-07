/**
 * The README is the single source of truth for configuration (scripts/build-skill.mjs copies its
 * marked regions into the companion skill; hindsight-docs/scripts/sync-coding-agents-doc.mjs copies
 * the whole thing into the docs site). These tests keep that generation honest: the committed skill
 * is the one the current README produces, it carries the configuration reference rather than a
 * prose subset of it, and region extraction refuses markup that would silently truncate a section
 * or merge it into the one above.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
// @ts-expect-error — plain .mjs build script, no type declarations
import { regions } from "../scripts/build-skill.mjs";

const pkgRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

describe("companion skill", () => {
  it("is up to date with the README", () => {
    // Throws with the regeneration command in its message when the skill is stale.
    execFileSync("node", [join(pkgRoot, "scripts", "build-skill.mjs"), "--check"], {
      cwd: pkgRoot,
      stdio: "pipe",
    });
  });

  it("carries the configuration reference, not a subset of it", () => {
    const skill = readFileSync(join(pkgRoot, "skill", "SKILL.md"), "utf8");
    // Bank-routing keys (the failure #3735 reported: one bank shared across unrelated projects)
    // and the ownership keys that decide which memory an agent writes to at all.
    for (const key of [
      "bankIdTemplate",
      "dynamicBankId",
      "mapPathToBank",
      "disabled",
      "bank",
      "principals",
      "principal",
    ]) {
      expect(skill).toContain(`\`${key}\``);
    }
  });
});

describe("skill region extraction", () => {
  const extract = (src: string): string[] => regions(src) as string[];

  it("normalises a region's headings so its top level becomes the skill's ##", () => {
    const [out] = extract(
      [
        "<!-- skill:begin -->",
        "### Reference",
        "",
        "#### Recipe",
        "",
        "text",
        "<!-- skill:end -->",
      ].join("\n")
    );
    expect(out).toContain("## Reference");
    expect(out).toContain("### Recipe");
  });

  it("keeps a region that already starts at ## unchanged", () => {
    const [out] = extract(
      ["<!-- skill:begin -->", "## Configuration", "", "### Opt-in", "<!-- skill:end -->"].join(
        "\n"
      )
    );
    expect(out).toContain("## Configuration");
    expect(out).toContain("### Opt-in");
  });

  it("synthesises a heading for a region that starts mid-section", () => {
    const [out] = extract(
      ['<!-- skill:begin title="Install / update" -->', "run it", "<!-- skill:end -->"].join("\n")
    );
    expect(out).toBe("## Install / update\n\nrun it");
  });

  it("refuses a headingless region with no title rather than silently merging it upward", () => {
    expect(() =>
      extract(["<!-- skill:begin -->", "orphan text", "<!-- skill:end -->"].join("\n"))
    ).toThrow(/must declare title=/);
  });

  it("refuses an unterminated region rather than swallowing the rest of the README", () => {
    expect(() => extract(["<!-- skill:begin -->", "## A", "text"].join("\n"))).toThrow(
      /unterminated/
    );
  });

  it("refuses a README with no marked regions at all", () => {
    expect(() => extract("## Configuration\n\nno markers here")).toThrow(/no <!-- skill:begin -->/);
  });
});
