import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

# Ensure workspace root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scratch.mp_ocr_worker import run_worker_chunk

UI_WORDS = {
    'bumble', 'profile', 'discover', 'people', 'liked you', 'chats', 'c;',
    'c:', 'c', 'things we can talk about', 'we have things in common', 'my bio',
    'swipe right', 'swipe left', 'what a match', 'send a compliment',
    'photo verified', 'verified'
}

ZODIAC = {
    'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
    'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces'
}

EDUCATION_BADGES = {
    'undergraduate degree', 'in college', 'postgraduate degree',
    'high school', 'trade school', 'doctorate'
}

def parse_lines(lines):
    name = None
    age = None
    verified = False
    badges = []
    jobs = []
    schools = []
    bio = []
    distance_km = None

    for line in lines:
        raw = line.strip()
        # Strip leading icons/bullets
        clean = re.sub(r'^[^\w\s]+', '', raw).strip()
        # Clean non-ascii
        clean = re.sub(r'[^\x20-\x7E]+', ' ', clean).strip()
        clean = re.sub(r'\s+', ' ', clean)
        low = clean.lower()

        if not clean or low in UI_WORDS or len(clean) <= 1:
            continue

        if 'photo verified' in low or 'verified' in low:
            verified = True
            continue
        if 'new here' in low:
            badges.append('New here')
            continue
        if 'travel mode' in low:
            badges.append('Travel mode')
            continue
        if any(z == low or f' {z} ' in f' {low} ' for z in ZODIAC):
            badges.append(clean)
            continue
        if any(eb in low for eb in EDUCATION_BADGES):
            badges.append(clean)
            continue

        m_dist = re.search(r'(\d+)\s*(?:km|miles|kilometer)', low)
        if m_dist:
            distance_km = int(m_dist.group(1))
            continue

        # Match Name & Age
        if name is None:
            # Pattern 1: Exact Name, 25 or Name 25
            m = re.search(r'^([A-Za-z\s\.\'\-]{2,25})[,\s]+(\d{2})$', clean)
            if m:
                cand_age = int(m.group(2))
                if 18 <= cand_age <= 65:
                    name = m.group(1).strip()
                    age = cand_age
                    continue

            # Pattern 2: Substring Name 25 inside line
            m_sub = re.search(r'\b([A-Za-z]{2,15})[,\s]+(\d{2})\b', clean)
            if m_sub:
                cand_age = int(m_sub.group(2))
                if 18 <= cand_age <= 65:
                    name = m_sub.group(1).strip()
                    age = cand_age
                    continue

        # University / Education
        if any(w in low for w in [
            'university', 'universitas', 'ugm', 'ui', 'itb', 'binus', 'unj', 'usu',
            'unpad', 'undip', 'unair', 'stikes', 'alumni', 'graduated', 'trisakti',
            'atma jaya', 'brawijaya', 'politeknik', 'ipb', 'upn', 'unpar', 'umn',
            '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
            '2019', '2020', '2021', '2022', '2023', '2024', '2025'
        ]):
            schools.append(clean)
            continue

        # Occupation / Job
        if any(w in low for w in [
            'manager', 'staff', 'engineer', 'nurse', 'doctor', 'officer', 'consultant',
            'founder', 'ceo', 'director', 'specialist', 'analyst', 'marketing', 'teacher',
            'lecturer', 'civil servant', 'pns', 'bumn', 'bank', 'banker', 'lawyer', 'notary',
            'architect', 'designer', 'dentist', 'pharmacist', 'journalist', 'writer',
            'editor', 'freelance', 'business', 'entrepreneur', 'hr', 'recruiter',
            'accountant', 'accounting', 'finance', 'creative', 'supervisor', 'executive',
            'associate', 'intern', 'operations', 'sales', 'lead', 'head of', 'co-founder'
        ]):
            jobs.append(clean)
            continue

        bio.append(clean)

    return {
        'name': name,
        'age': age,
        'verified': verified,
        'badge': ' | '.join(badges) if badges else None,
        'job': ' | '.join(jobs) if jobs else None,
        'school': ' | '.join(schools) if schools else None,
        'distance_km': distance_km,
        'bio_text': ' | '.join(bio) if bio else None,
        'raw_text': ' | '.join(lines) if lines else ''
    }


def parse_timestamp_from_filename(fname: str):
    m = re.search(r'profile_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', fname)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}"
    return None


def main():
    print("=" * 80)
    print("EXTRACTING FULL 54K PROFILE METADATA & BIOS USING HIGH-THROUGHPUT OCR")
    print("=" * 80)

    bumble_dir = Path(r"D:\BumbleLog")
    cropped_dir = Path(r"D:\BumbleLog\cropped")
    csv_labels = Path(r"D:\BumbleLog\binary_swipe_labels.csv")
    out_csv = Path(r"D:\BumbleLog\bumble_54k_profile_metadata.csv")

    # 1. Load Labels Map
    labels_map = {}
    if csv_labels.exists():
        df_lbl = pd.read_csv(csv_labels)
        for _, row in df_lbl.iterrows():
            fname = os.path.basename(str(row['path']))
            labels_map[fname] = {
                'rating': float(row['rating']),
                'action': 'right' if float(row['rating']) >= 50.0 else 'left'
            }
        print(f"Loaded {len(labels_map)} labels from {csv_labels}")

    # 2. Collect all raw images
    all_files = sorted(list(bumble_dir.glob("*.webp")) + list(bumble_dir.glob("*.jpg")) + list(bumble_dir.glob("*.png")))
    total_files = len(all_files)
    print(f"Found {total_files} total raw screenshots in {bumble_dir}")

    # 3. Resume from existing CSV if present
    existing_records = []
    seen_filenames = set()
    if out_csv.exists():
        try:
            df_exist = pd.read_csv(out_csv)
            seen_filenames = set(df_exist['filename'].dropna())
            existing_records = df_exist.to_dict('records')
            print(f"Found existing CSV with {len(existing_records)} records! Resuming from index {len(existing_records)}...")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    remaining_files = [f for f in all_files if f.name not in seen_filenames]
    print(f"Remaining files to process: {len(remaining_files)}")

    if not remaining_files:
        print("All 54k files already processed!")
        return

    n_workers = 6
    chunk_size = 1200 # 200 per worker per batch
    all_records = existing_records

    t_start = time.time()
    n_processed = 0

    pbar = tqdm(total=len(remaining_files), desc="Processing 54k Profiles")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in range(0, len(remaining_files), chunk_size):
            batch_files = remaining_files[i : i + chunk_size]
            
            # Split batch among workers
            sub_chunk_size = (len(batch_files) + n_workers - 1) // n_workers
            sub_chunks = []
            for w in range(n_workers):
                sub = batch_files[w * sub_chunk_size : (w + 1) * sub_chunk_size]
                if sub:
                    sub_chunks.append([str(f.resolve()) for f in sub])

            # Run parallel OCR workers
            worker_results = []
            for res in executor.map(run_worker_chunk, sub_chunks):
                worker_results.extend(res)

            # Map results to records
            batch_records = []
            for fname, lines in worker_results:
                parsed = parse_lines(lines)
                ts = parse_timestamp_from_filename(fname)
                lbl = labels_map.get(fname, None)
                rating = lbl['rating'] if lbl is not None else None
                action = lbl['action'] if lbl is not None else 'unlabeled'
                is_labeled = lbl is not None

                crop_p = cropped_dir / (Path(fname).stem + ".webp")
                cropped_path = str(crop_p) if crop_p.exists() else None

                batch_records.append({
                    'filename': fname,
                    'raw_path': str(bumble_dir / fname),
                    'cropped_path': cropped_path,
                    'timestamp': ts,
                    'is_labeled': is_labeled,
                    'rating': rating,
                    'action': action,
                    'name': parsed['name'],
                    'age': parsed['age'],
                    'verified': parsed['verified'],
                    'badge': parsed['badge'],
                    'job': parsed['job'],
                    'school': parsed['school'],
                    'distance_km': parsed['distance_km'],
                    'bio_text': parsed['bio_text'],
                    'raw_text': parsed['raw_text']
                })

            all_records.extend(batch_records)
            n_processed += len(batch_records)
            pbar.update(len(batch_records))

            # Periodic save to disk every 1,200 profiles
            df_temp = pd.DataFrame(all_records)
            df_temp.to_csv(out_csv, index=False)

            elapsed = time.time() - t_start
            speed = n_processed / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining_files) - n_processed) / speed / 60 if speed > 0 else 0

            n_names = df_temp['name'].notna().sum()
            n_ages = df_temp['age'].notna().sum()
            n_jobs = df_temp['job'].notna().sum()
            n_schools = df_temp['school'].notna().sum()
            n_bios = df_temp['bio_text'].notna().sum()
            n_ver = df_temp['verified'].sum()

            pbar.set_postfix({
                "Speed": f"{speed:.1f}/s",
                "ETA": f"{eta_min:.1f}m",
                "Names": f"{n_names}",
                "Ages": f"{n_ages}",
                "Jobs": f"{n_jobs}",
                "Schools": f"{n_schools}",
                "Bios": f"{n_bios}",
                "Ver": f"{n_ver}"
            })

    pbar.close()

    df_final = pd.DataFrame(all_records)
    df_final.to_csv(out_csv, index=False)
    print(f"\n[COMPLETE] Successfully extracted and saved {len(df_final)} profile metadata records to:")
    print(f"  --> {out_csv}")

    # Generate 500-sample preview for artifacts
    artifact_summary = Path(r"C:\Users\Rivian\.gemini\antigravity\brain\73824904-181e-4277-ac0d-85aa19274479") / "bumble_profile_metadata_sample.csv"
    df_final.head(500).to_csv(artifact_summary, index=False)
    print(f"Saved 500-sample preview to {artifact_summary}")


if __name__ == "__main__":
    main()
