import asyncio
from pathlib import Path
import winsdk.windows.media.ocr as ocr
import winsdk.windows.graphics.imaging as imaging
import winsdk.windows.storage as storage

def run_worker_chunk(file_paths):
    async def _async_run():
        engine = ocr.OcrEngine.try_create_from_user_profile_languages()
        results = []
        for p_str in file_paths:
            try:
                p = Path(p_str)
                f_storage = await storage.StorageFile.get_file_from_path_async(str(p.resolve()))
                stream = await f_storage.open_async(storage.FileAccessMode.READ)
                decoder = await imaging.BitmapDecoder.create_async(stream)
                bitmap = await decoder.get_software_bitmap_async()
                res = await engine.recognize_async(bitmap)
                lines = [l.text for l in res.lines] if res and res.lines else []
                results.append((p.name, lines))
            except Exception as e:
                results.append((Path(p_str).name, []))
        return results
    return asyncio.run(_async_run())
