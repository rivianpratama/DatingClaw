import asyncio
import time
from pathlib import Path
import winsdk.windows.media.ocr as ocr
import winsdk.windows.graphics.imaging as imaging
import winsdk.windows.storage as storage

async def bench(sem_count, n_files=100):
    engine = ocr.OcrEngine.try_create_from_user_profile_languages()
    files = list(Path(r"D:\BumbleLog").glob("*.webp"))[100:100+n_files]
    sem = asyncio.Semaphore(sem_count)

    async def worker(f):
        async with sem:
            try:
                f_storage = await storage.StorageFile.get_file_from_path_async(str(f.resolve()))
                stream = await f_storage.open_async(storage.FileAccessMode.READ)
                decoder = await imaging.BitmapDecoder.create_async(stream)
                bitmap = await decoder.get_software_bitmap_async()
                res = await engine.recognize_async(bitmap)
                return len(res.lines) if res and res.lines else 0
            except Exception as e:
                return -1

    t0 = time.time()
    results = await asyncio.gather(*[worker(f) for f in files])
    elapsed = time.time() - t0
    success = sum(1 for r in results if r >= 0)
    fps = n_files / elapsed
    print(f"Sem {sem_count}: {success}/{n_files} success in {elapsed:.2f}s ({fps:.1f} fps)")

async def main():
    for s in [2, 4, 8]:
        await bench(s, 100)

if __name__ == "__main__":
    asyncio.run(main())
