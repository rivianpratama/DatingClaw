import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    
    const file = formData.get('image') as File | null;
    const modelVersion = formData.get('model_version') as string || 'bumble_10k_pure';

    if (!file) {
      return NextResponse.json({ error: 'No image file uploaded' }, { status: 400 });
    }

    const pyFormData = new FormData();
    pyFormData.append('image', file);
    pyFormData.append('model_version', modelVersion);

    const pyPort = process.env.BUMBLECLAW_PORT || '7860';
    const pyUrl = `http://127.0.0.1:${pyPort}/analyze`;

    const pyRes = await fetch(pyUrl, {
      method: 'POST',
      body: pyFormData,
    });

    if (!pyRes.ok) {
      const errorText = await pyRes.text();
      let parsedError = errorText;
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.detail || errJson.error || errorText;
      } catch {}
      return NextResponse.json({ error: `Python API Error: ${parsedError}` }, { status: pyRes.status });
    }

    const data = await pyRes.json();
    return NextResponse.json(data);
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Internal Server Error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
