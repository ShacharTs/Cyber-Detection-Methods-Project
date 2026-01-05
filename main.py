import os
import sys
import subprocess
import pandas as pd

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Malware split outputs (ASSUMED PRE-BUILT)
MAL_TRAIN = os.path.join(OUT_DIR, "malware_train.csv")
MAL_VALID = os.path.join(OUT_DIR, "malware_valid.csv")
MAL_TEST  = os.path.join(OUT_DIR, "malware_test.csv")

# Benign split outputs
BEN_TRAIN = os.path.join(OUT_DIR, "non_malware_train.csv")
BEN_VALID = os.path.join(OUT_DIR, "non_malware_valid.csv")
BEN_TEST  = os.path.join(OUT_DIR, "non_malware_test.csv")

# Final merged outputs
FINAL_TRAIN = os.path.join(OUT_DIR, "npm_train.csv")
FINAL_VALID = os.path.join(OUT_DIR, "npm_valid.csv")
FINAL_TEST  = os.path.join(OUT_DIR, "npm_test.csv")

# Scripts
BEN_SCRIPT = os.path.join(BASE_DIR, "load_non_malware.py")


# =========================================================
# HELPERS
# =========================================================
def run_script(script_path: str, extra_env: dict | None = None):
    print(f"[+] Running {os.path.basename(script_path)}")
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(script_path)} failed (exit={result.returncode})"
        )


def exists_and_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def exists_all(paths: list[str]) -> bool:
    return all(exists_and_nonempty(p) for p in paths)


def merge_split(mal_path: str, ben_path: str, out_path: str):
    df_m = pd.read_csv(mal_path)
    df_b = pd.read_csv(ben_path)

    df = pd.concat([df_m, df_b], ignore_index=True)

    # Best-effort deduplication
    if "package_name" in df.columns:
        df = df.drop_duplicates(subset=["package_name"], keep="first")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    if "label" in df.columns:
        vc = df["label"].value_counts().to_dict()
    else:
        vc = {}

    print(
        f"[✓] {os.path.basename(out_path)} "
        f"rows={len(df)} labels={vc}"
    )


# =========================================================
# MAIN
# =========================================================
def main():
    # -----------------------------------------------------
    # 1) Malware MUST already exist (never rebuilt here)
    # -----------------------------------------------------
    if not exists_all([MAL_TRAIN, MAL_VALID, MAL_TEST]):
        print("[ERROR] Malware CSVs are missing.")
        print("This pipeline assumes malware is already built.")
        sys.exit(1)

    print("[✓] Malware CSVs detected — skipping malware generation")

    # -----------------------------------------------------
    # 2) Count malware sizes (authoritative)
    # -----------------------------------------------------
    mal_train_n = len(pd.read_csv(MAL_TRAIN))
    mal_valid_n = len(pd.read_csv(MAL_VALID))
    mal_test_n  = len(pd.read_csv(MAL_TEST))

    print(
        f"[i] Malware sizes → "
        f"train={mal_train_n}, valid={mal_valid_n}, test={mal_test_n}"
    )

    # -----------------------------------------------------
    # 3) Build benign ONLY if missing
    # -----------------------------------------------------
    need_benign = not exists_all([BEN_TRAIN, BEN_VALID, BEN_TEST])

    if need_benign:
        print("[+] Benign CSVs missing — generating non-malware data")
        run_script(
            BEN_SCRIPT,
            extra_env={
                "TARGET_TRAIN": mal_train_n,
                "TARGET_VALID": mal_valid_n,
                "TARGET_TEST":  mal_test_n,
            },
        )
    else:
        print("[✓] Benign CSVs already exist — skipping benign generation")

    if not exists_all([BEN_TRAIN, BEN_VALID, BEN_TEST]):
        print("[ERROR] Non-malware CSVs missing after generation.")
        sys.exit(1)

    # -----------------------------------------------------
    # 4) Merge splits
    # -----------------------------------------------------
    print("[+] Merging datasets")
    merge_split(MAL_TRAIN, BEN_TRAIN, FINAL_TRAIN)
    merge_split(MAL_VALID, BEN_VALID, FINAL_VALID)
    merge_split(MAL_TEST,  BEN_TEST,  FINAL_TEST)

    print("\n===== SUCCESS =====")
    print(f"Train: {FINAL_TRAIN}")
    print(f"Valid: {FINAL_VALID}")
    print(f"Test : {FINAL_TEST}")


if __name__ == "__main__":
    main()
