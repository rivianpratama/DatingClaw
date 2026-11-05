import { NextResponse } from 'next/server';
import { constants, createReadStream } from 'node:fs';
import fs from 'node:fs';
import { access } from 'node:fs/promises';
import { Readable } from 'node:stream';
import path from 'path';
import { getBumbleLogRoot, getBumbleRunFilePath } from '@/lib/bumbleLog';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const file = searchParams.get('file');
  const run = searchParams.get('run');

  if (!file) {
    return new NextResponse('File missing', { status: 400 });
  }

  const safeFile = path.basename(file);
  const resolveImagePath = () => {
    const primary = getBumbleRunFilePath(safeFile, run);
    if (fs.existsSync(primary)) return primary;
    const root = getBumbleLogRoot();
    const candidates = [root];
    try {
      for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        if (entry.isDirectory()) candidates.push(path.join(root, entry.name));
      }
    } catch { /* ignore */ }
    return candidates.map((dir) => path.join(dir, safeFile)).find((candidate) => fs.existsSync(candidate)) || primary;
  };

  let filePath = resolveImagePath();
  // The runner writes the CSV row before the async WebP save completes. Wait briefly.
  const deadline = Date.now() + 1500;
  while (!fs.existsSync(filePath) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
    filePath = resolveImagePath();
  }

  try {
    try {
      await access(filePath, constants.R_OK);
    } catch {
      return new NextResponse('File not found', { status: 404 });
    }

    const ext = path.extname(safeFile).toLowerCase();
    
    let contentType = 'image/jpeg';
    if (ext === '.webp') contentType = 'image/webp';
    else if (ext === '.png') contentType = 'image/png';
    else if (ext === '.gif') contentType = 'image/gif';

    const stream = Readable.toWeb(createReadStream(filePath)) as ReadableStream;

    return new NextResponse(stream, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-store, max-age=0',
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return new NextResponse(message, { status: 500 });
  }
}
