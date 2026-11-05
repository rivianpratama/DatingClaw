import asyncio
import re
from pathlib import Path
import winsdk.windows.media.ocr as ocr
import winsdk.windows.graphics.imaging as imaging
import winsdk.windows.storage as storage

UI_WORDS = {
    'bumble', 'profile', 'discover', 'people', 'liked you', 'chats', 'c;',
    'c:', 'c', 'things we can talk about', 'we have things in common', 'my bio'
}

def parse_profile_text(lines):
    name = None
    age = None
    verified = False
    badge = None
    distance_km = None
    bio_lines = []

    for l in lines:
        cleaned = l.strip()
        low = cleaned.lower()
        if not cleaned:
            continue
        if low in UI_WORDS:
            continue
        if 'photo verified' in low or 'verified' in low:
            verified = True
            continue
        if 'new here' in low:
            badge = 'New here'
            continue
        if 'travel mode' in low:
            badge = 'Travel mode'
            continue

        # Distance: e.g. '5 km away', '12 miles away', 'Nearby'
        m_dist = re.search(r'(\d+)\s*(?:km|miles|kilometer)', low)
        if m_dist:
            distance_km = int(m_dist.group(1))
            continue

        # Name and Age: e.g. 'Sarah, 24' or 'Jessica 25' or 'Ed, 26'
        m_name_age = re.search(r'^([A-Za-z\s\.\'\-]+)[,\s]+(\d{2})$', cleaned)
        if m_name_age and name is None:
            name = m_name_age.group(1).strip()
            try:
                age = int(m_name_age.group(2))
                continue
            except:
                pass
        
        # Fallback name and age search within line
        m_sub = re.search(r'([A-Za-z]{2,15})[,\s]+(\d{2})\b', cleaned)
        if m_sub and name is None:
            name = m_sub.group(1).strip()
            try:
                age = int(m_sub.group(2))
                continue
            except:
                pass

        # Otherwise treat as bio / info line
        bio_lines.append(cleaned)
    
    return {
        'name': name,
        'age': age,
        'verified': verified,
        'badge': badge,
        'distance_km': distance_km,
        'bio': ' | '.join(bio_lines),
        'raw_lines': ' | '.join(lines)
    }

async def main():
    engine = ocr.OcrEngine.try_create_from_user_profile_languages()
    bumble_dir = Path(r'D:\BumbleLog')
    samples = list(bumble_dir.glob('*.webp'))[:10]
    print(f"Testing OCR parsing on {len(samples)} sample images:\n")
    for s in samples:
        try:
            file = await storage.StorageFile.get_file_from_path_async(str(s.resolve()))
            stream = await file.open_async(storage.FileAccessMode.READ)
            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            res = await engine.recognize_async(bitmap)
            p = parse_profile_text([l.text for l in res.lines])
            print(f"File: {s.name}")
            print(f"  Name: {p['name']} | Age: {p['age']} | Verified: {p['verified']} | Badge: {p['badge']} | Dist: {p['distance_km']}")
            print(f"  Bio: {p['bio']}")
            print("-" * 60)
        except Exception as e:
            print(f"Error on {s.name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
