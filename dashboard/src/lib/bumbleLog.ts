import fs from 'node:fs';
import path from 'node:path';

const DEFAULT_LOG_DRIVE = ['D', ':'].join('');
const DEFAULT_LOG_FOLDER = 'BumbleLog';

export function getBumbleLogRoot() {
  return process.env.BUMBLE_LOG_DIR || path.join(`${DEFAULT_LOG_DRIVE}${path.sep}`, DEFAULT_LOG_FOLDER);
}

function newestRunId(root: string): string | null {
  try {
    let best: string | null = null;
    let bestMtime = -Infinity;
    // Prefer a real run subfolder over the root aggregate scores.csv.
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const scores = path.join(root, entry.name, 'scores.csv');
      if (!fs.existsSync(scores)) continue;
      const mtime = fs.statSync(scores).mtimeMs;
      if (mtime > bestMtime) { best = entry.name; bestMtime = mtime; }
    }
    return best;
  } catch { return null; }
}

/** A run lives in a subdirectory with its own scores.csv (e.g. untrained_test_2000). */
export function getBumbleRunDir(run?: string | null) {
  if (run) return path.join(getBumbleLogRoot(), path.basename(run));
  if (process.env.BUMBLE_RUN_DIR) return path.resolve(process.env.BUMBLE_RUN_DIR);
  const root = getBumbleLogRoot();
  const newest = newestRunId(root);
  return newest ? path.join(root, newest) : root;
}

export function getBumbleRunFilePath(file: string, run?: string | null) {
  return path.join(getBumbleRunDir(run), path.basename(file));
}

/** Backward-compatible default: root of BumbleLog. */
export function getBumbleLogDir() {
  return getBumbleLogRoot();
}

export function getBumbleLogFilePath(file: string) {
  return getBumbleRunFilePath(file);
}
