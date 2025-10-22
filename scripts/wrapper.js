#!/usr/bin/env node

/**
 * wrapper.js
 * -----------
 * Wrapper for running an Artillery load test while periodically writing
 * intermediate JSON stats to a file.
 *
 * Usage:
 *   node wrapper.js <scenario_file.yml> <final_json_path> <latest_stats_json_path>
 *
 * Arguments:
 *   scenario_file.yml       Path to the Artillery YAML scenario file
 *   final_json_path         Path to write final JSON results
 *   latest_stats_json_path  Path to write intermediate stats periodically
 */

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

// --- Parse command-line arguments ---
if (process.argv.length < 5) {
  console.error("Usage: node wrapper.js <scenario.yml> <final.json> <latest_stats.json>");
  process.exit(1);
}

const scenarioFile = process.argv[2];
const finalJson = process.argv[3];
const intermediateJson = process.argv[4];

// --- Ensure scenario file exists ---
if (!fs.existsSync(scenarioFile)) {
  console.error(`Scenario file not found: ${scenarioFile}`);
  process.exit(1);
}

// --- Temp file to hold intermediate Artillery output ---
const tmpFile = path.join(path.dirname(intermediateJson), `tmp_${Date.now()}.json`);

// --- Spawn Artillery as a child process ---
const artillery = spawn("npx", [
  "artillery",
  "run",
  scenarioFile,
  "--output",
  tmpFile,
]);

console.log(`Running Artillery test: ${scenarioFile}`);
console.log(`Intermediate stats will be written to: ${intermediateJson}`);
console.log(`Final JSON will be written to: ${finalJson}`);

// --- Function to copy tmp JSON to intermediate JSON periodically ---
let lastStatsUpdate = 0;
const intervalMs = 5000; // Update every 5 seconds
const interval = setInterval(() => {
  if (fs.existsSync(tmpFile)) {
    try {
      const data = fs.readFileSync(tmpFile, "utf8");
      fs.writeFileSync(intermediateJson, data);
      lastStatsUpdate = Date.now();
      console.log(`Updated intermediate stats at ${new Date().toISOString()}`);
    } catch (err) {
      console.warn("Could not update intermediate stats:", err.message);
    }
  }
}, intervalMs);

// --- Forward stdout/stderr from Artillery to this process ---
artillery.stdout.on("data", (data) => process.stdout.write(data));
artillery.stderr.on("data", (data) => process.stderr.write(data));

// --- Handle process exit ---
artillery.on("close", (code) => {
  clearInterval(interval);

  if (fs.existsSync(tmpFile)) {
    fs.renameSync(tmpFile, finalJson); // Save final JSON
    console.log(`Final JSON written to: ${finalJson}`);
  } else {
    console.error("Artillery did not produce output JSON.");
  }

  process.exit(code);
});
