import pandas as pd
from pathlib import Path

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "output"

NON_MALWARE_CSV = INPUT_DIR / "npm_non_malware.csv"
MALWARE_CSV     = INPUT_DIR / "npm_malware.csv"

OUT_DIR = BASE_DIR / "output_csv"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV  = OUT_DIR / "test.csv"

TEST_RATIO = 0.20
RANDOM_SEED = 42

# =========================
# 1. LOAD & LABEL
# =========================
# Read the files
df_non = pd.read_csv(NON_MALWARE_CSV)
df_mal = pd.read_csv(MALWARE_CSV)

# Clean existing labels if they exist and set fresh ones
if "label" in df_non.columns:
    df_non = df_non.drop(columns=["label"])
if "label" in df_mal.columns:
    df_mal = df_mal.drop(columns=["label"])

df_non["label"] = 0  # non-malware
df_mal["label"] = 1  # malware

print(f"[+] Loaded non-malware: {len(df_non)}")
print(f"[+] Loaded malware    : {len(df_mal)}")

# =========================
# 2. MERGE & SHUFFLE
# =========================
# ignore_index=True prevents duplicate index issues
full_df = pd.concat([df_non, df_mal], ignore_index=True)

# Shuffle the entire dataset so malware and non-malware are mixed
full_df = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# =========================
# 3. CALCULATE SPLIT
# =========================
# We want 80% for train, so the split point is total * 0.8
total_rows = len(full_df)
split_point = int(total_rows * (1 - TEST_RATIO))

# Split the data using integer-based slicing (iloc)
train_df = full_df.iloc[:split_point]
test_df  = full_df.iloc[split_point:]

# =========================
# 4. VERIFY & SAVE
# =========================
print("\n[COUNTS]")
print(f"Total rows: {total_rows}")
print(f"Train rows: {len(train_df)} ({len(train_df)/total_rows:.0%})")
print(f"Test rows : {len(test_df)} ({len(test_df)/total_rows:.0%})")

print("\n[LABEL DISTRIBUTION IN TRAIN]")
print(train_df["label"].value_counts())

# Save to CSV
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n[✓] DONE: Files saved in 'output_csv/'")