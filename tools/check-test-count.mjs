#!/usr/bin/env node
/**
 * Fail when the README's "Tests: N" line disagrees with the suite.
 *
 * Rule from the handover: a claim not backed by a test is unverified, and a count
 * that drifts from the suite is the cheapest way for a README to become dishonest.
 * This runs the suite, reads the real number, and refuses to pass on a mismatch.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "test-count-"));
const report = join(dir, "report.json");
let total;
try {
  execFileSync("npx", ["vitest", "run", "--reporter=json", `--outputFile=${report}`], { stdio: "pipe" });
  total = JSON.parse(readFileSync(report, "utf8")).numTotalTests;
} finally {
  rmSync(dir, { recursive: true, force: true });
}

const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
const match = readme.match(/^\*\*Tests:\*\* (\d+)/m) ?? readme.match(/^Tests: (\d+)/m);
if (!match) {
  console.error('no "Tests: N" line in README.md');
  process.exit(1);
}
const claimed = Number(match[1]);
if (claimed !== total) {
  console.error(`README claims ${claimed} tests, suite has ${total}. Update the README (or the tests).`);
  process.exit(1);
}
console.log(`test count matches README: ${total}`);
