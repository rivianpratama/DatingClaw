import { NextResponse } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { getBumbleLogRoot } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

type RunInfo = { id: string; name: string; count: number; mtime: number };

function countRows(file: string) {
  try {
    const text = fs.readFileSync(file, 'utf-8').trim();
    if (!text) return 0;
    return Math.max(0, text.split(/\r?\n/).length - 1);
  } catch {
    return 0;
  }
}

export async function GET() {
  const root = getBumbleLogRoot();
  const runs: RunInfo[] = [];
  try {
    const rootScores = path.join(root, 'scores.csv');
    if (fs.existsSync(rootScores)) {
      const stat = fs.statSync(rootScores);
      runs.push({ id: '.', name: 'root', count: countRows(rootScores), mtime: stat.mtimeMs });
    }
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const scores = path.join(root, entry.name, 'scores.csv');
      if (!fs.existsSync(scores)) continue;
      const stat = fs.statSync(scores);
      runs.push({ id: entry.name, name: entry.name, count: countRows(scores), mtime: stat.mtimeMs });
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message, runs: [] }, { status: 500 });
  }
  runs.sort((a, b) => b.mtime - a.mtime);
  return NextResponse.json({ runs });
}
