import os
import zipfile
import json
import csv
import shutil
import pandas as pd

# --- הגדרות נתיבים ---
root_path = os.path.expanduser("~/datadog_compromised/samples/npm")
output_csv = "npm_malicious_features.csv"
zip_password = b"infected"
temp_extract_dir = "./temp_extract"

# רשימת העמודות להשוואה (ללא השם והתווית)
feature_columns = [
    "num_dependencies", "num_dev_dependencies", "has_scripts",
    "description_length", "repository_exists", "license_exists", "num_js_files",
    "package_name_length", "has_bin", "has_install_script", "num_install_scripts",
    "num_maintainers", "many_maintainers_flag", "num_contributors",
    "maintainer_to_contributor_ratio", "days_since_last_update", "inactive_package_flag"
]


# --- שלב 1: חילוץ נתונים גולמיים מה-ZIP ---
def stage_1_extract_raw_data():
    """Extracts features from every ZIP and saves to a raw CSV."""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
        header = ["package_name"] + feature_columns + ["label"]
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()

        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    try:
                        if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
                        os.makedirs(temp_extract_dir)
                        with zipfile.ZipFile(zip_path) as z:
                            target_json = next((f for f in z.namelist() if f.endswith("package.json")), None)
                            if target_json:
                                z.extract(target_json, path=temp_extract_dir, pwd=zip_password)
                                with open(os.path.join(temp_extract_dir, target_json), 'r') as jf:
                                    data = json.load(jf)

                                    # Extract Logic
                                    scripts = data.get("scripts", {})
                                    inst_keys = ["install", "preinstall", "postinstall"]
                                    found_inst = [s for s in scripts if s in inst_keys]
                                    num_m = len(data.get("maintainers", [])) if isinstance(data.get("maintainers"),
                                                                                           list) else 1
                                    num_c = len(data.get("contributors", [])) if isinstance(data.get("contributors"),
                                                                                            list) else 0

                                    pkg_real_name = str(data.get("name", "N/A")).strip()
                                    pkg_version = str(data.get("version", "0.0.0")).strip()

                                    writer.writerow({
                                        "package_name": f"{pkg_real_name}@{pkg_version}",
                                        "num_dependencies": len(data.get("dependencies", {})),
                                        "num_dev_dependencies": len(data.get("devDependencies", {})),
                                        "has_scripts": 1 if scripts else 0,
                                        "description_length": len(data.get("description", "") or ""),
                                        "repository_exists": 1 if data.get("repository") else 0,
                                        "license_exists": 1 if data.get("license") else 0,
                                        "num_js_files": len([f for f in z.namelist() if f.endswith('.js')]),
                                        "package_name_length": len(pkg_real_name),
                                        "has_bin": 1 if data.get("bin") else 0,
                                        "has_install_script": 1 if found_inst else 0,
                                        "num_install_scripts": len(found_inst),
                                        "num_maintainers": num_m,
                                        "many_maintainers_flag": 1 if num_m > 10 else 0,
                                        "num_contributors": num_c,
                                        "maintainer_to_contributor_ratio": round(num_m / num_c,
                                                                                 2) if num_c > 0 else float(num_m),
                                        "days_since_last_update": 0,
                                        "inactive_package_flag": 0,
                                        "label": 1
                                    })
                        shutil.rmtree(temp_extract_dir)
                    except Exception:
                        continue
    print("Stage 1: Raw data extraction complete.")


# --- שלב 2: ניקוי כפילויות חכם (Smart Deduplication) ---
def stage_2_smart_cleanup():
    """Removes rows only if the package name and ALL features are identical."""
    if not os.path.exists(output_csv): return

    df = pd.read_csv(output_csv)
    initial_count = len(df)

    # יצירת שם בסיס ללא גרסה לצורך השוואה לוגית
    df['base_name'] = df['package_name'].str.split('@').str[0]

    # הליבה של הלוגיקה שלך:
    # מסירים כפילויות רק אם ה-base_name וכל הפיצ'רים זהים ב-100%
    df_unique = df.drop_duplicates(subset=['base_name'] + feature_columns, keep='first')

    # מחיקת עמודת העזר ושמירה
    df_final = df_unique.drop(columns=['base_name']).reset_index(drop=True)
    df_final.to_csv(output_csv, index=False)

    print(f"Stage 2: Cleanup finished. Removed {initial_count - len(df_final)} redundant rows.")
    return df_final


# --- שלב 3: הפקת דו"ח סיכום (Optional Summary) ---
def stage_3_data_summary(df):
    """Prints a quick summary of the unique data found."""
    print("\n--- Data Summary ---")
    print(f"Total Unique Samples: {len(df)}")
    print(f"Average JS Files per Package: {df['num_js_files'].mean():.2f}")
    print(f"Packages with Install Scripts: {df['has_install_script'].sum()}")
    print("--------------------")


# --- הרצה מרכזית ---
if __name__ == "__main__":
    stage_1_extract_raw_data()
    cleaned_df = stage_2_smart_cleanup()
    if cleaned_df is not None:
        stage_3_data_summary(cleaned_df)