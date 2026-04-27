import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const frontendRoot = resolve(__dirname, "..");
const sourceDir = join(frontendRoot, "node_modules", "@mediapipe", "pose");
const targetDir = join(frontendRoot, "public", "mediapipe", "pose");

if (!existsSync(sourceDir)) {
  console.error(`[err] MediaPipe pose package not found: ${sourceDir}`);
  process.exit(1);
}

mkdirSync(targetDir, { recursive: true });

for (const entry of readdirSync(sourceDir)) {
  const src = join(sourceDir, entry);
  const dest = join(targetDir, entry);
  if (!statSync(src).isFile()) continue;
  cpSync(src, dest, { force: true });
}

console.log(`[ok] synced MediaPipe pose assets to ${targetDir}`);
