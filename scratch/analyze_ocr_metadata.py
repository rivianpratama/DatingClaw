import pandas as pd
import numpy as np

csv_path = r"D:\BumbleLog\bumble_54k_profile_metadata.csv"
df = pd.read_csv(csv_path, low_memory=False)

print(f"Total rows in metadata CSV: {len(df)}")
print("Columns:", df.columns.tolist())

# 1. Names
# Clean names: strip, title case, remove non-alphabetic artifacts if any
names = df['name'].dropna().astype(str).str.strip()
names = names[names.str.len() > 1] # filter single chars
names_clean = names.str.title()

print("\n=== Top 20 Most Common Names ===")
top_names = names_clean.value_counts().head(20)
print(top_names)

# 2. Occupations / Jobs
jobs = df['job'].dropna().astype(str).str.strip()
jobs = jobs[jobs.str.len() > 2]
# Filter out obvious OCR noise or UI text if any
top_jobs = jobs.value_counts().head(20)
print("\n=== Top 20 Most Common Occupations (Raw) ===")
print(top_jobs)

# Standardized job categories
print("\n=== Top Standardized Occupation Categories ===")
job_lower = jobs.str.lower()
job_cats = {}
categories = {
    "Doctor / Medical / Healthcare": ["doctor", "dokter", "dentist", "nurse", "perawat", "bidan", "pharmacist", "apoteker", "medical", "rsud", "koas"],
    "Software / Tech / IT / Engineer": ["software", "developer", "engineer", "data", "programmer", "tech", "it ", "frontend", "backend", "fullstack", "product manager", "analyst"],
    "Teacher / Lecturer / Educator": ["guru", "teacher", "dosen", "lecturer", "educator", "tutor"],
    "Finance / Accounting / Banking": ["finance", "accounting", "akuntan", "bank", "bca", "mandiri", "bri", "bni", "auditor", "tax", "pajak"],
    "Design / Creative / Architect": ["designer", "desain", "graphic", "architect", "arsitek", "art", "creative", "illustrator", "content creator"],
    "Marketing / Sales / Business Dev": ["marketing", "sales", "business development", "bizdev", "account executive", "brand"],
    "Law / Legal": ["lawyer", "advokat", "legal", "notaris", "hukum"],
    "Civil Servant / State / BUMN": ["pns", "asn", "bumn", "kemen", "pegawai negeri"],
    "Entrepreneur / Business Owner": ["owner", "founder", "entrepreneur", "wiraswasta", "business owner", "ceo"],
    "HR / Human Resources / Recruiter": ["hr", "human resources", "recruiter", "talent acquisition"],
}

for cat, kws in categories.items():
    mask = job_lower.apply(lambda x: any(kw in x for kw in kws))
    job_cats[cat] = int(mask.sum())

job_cats_series = pd.Series(job_cats).sort_values(ascending=False)
print(job_cats_series)

# 3. Universities / Schools
schools = df['school'].dropna().astype(str).str.strip()
schools = schools[schools.str.len() > 2]
print("\n=== Top 20 Most Common Universities (Raw) ===")
top_schools = schools.value_counts().head(20)
print(top_schools)

# Standardized university categories / aliases
print("\n=== Top Standardized Universities ===")
school_lower = schools.str.lower()
uni_patterns = {
    "Universitas Indonesia (UI)": ["universitas indonesia", "ui", "univ indonesia"],
    "Binus University (Bina Nusantara)": ["binus", "bina nusantara"],
    "Universitas Gadjah Mada (UGM)": ["gadjah mada", "ugm"],
    "Institut Teknologi Bandung (ITB)": ["itb", "institut teknologi bandung"],
    "Universitas Padjadjaran (Unpad)": ["unpad", "padjadjaran"],
    "Universitas Brawijaya (UB)": ["brawijaya", "ub"],
    "Universitas Trisakti": ["trisakti"],
    "Universitas Diponegoro (Undip)": ["undip", "diponegoro"],
    "Universitas Airlangga (Unair)": ["unair", "airlangga"],
    "Universitas Atma Jaya": ["atma jaya", "atmajaya"],
    "Universitas Tarumanagara (Untar)": ["untar", "tarumanagara"],
    "Universitas Sebelas Maret (UNS)": ["sebelas maret", "uns"],
    "Universitas Pelita Harapan (UPH)": ["uph", "pelita harapan"],
    "Universitas Telkom": ["telkom"],
    "Universitas Katolik Parahyangan (Unpar)": ["unpar", "parahyangan"],
    "Institut Pertanian Bogor (IPB)": ["ipb", "institut pertanian bogor"],
    "Universitas Sanata Dharma": ["sanata dharma"],
    "Universitas Prasetya Mulya (Prasmul)": ["prasetya mulya", "prasmul"],
}

uni_counts = {}
for name, aliases in uni_patterns.items():
    # word boundary match or clean substring
    mask = school_lower.apply(lambda x: any(alias in x for alias in aliases))
    uni_counts[name] = int(mask.sum())

uni_series = pd.Series(uni_counts).sort_values(ascending=False)
print(uni_series)
