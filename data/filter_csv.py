import pandas as pd

# 1. Define the Feature Lists (Updated)
# =========================
# Weak-Link
# =========================
WEAKLINK_FEATURES_LIST = [
    "num_dependencies",
    "num_dev_dependencies",
    "num_maintainers",
    "num_contributors",
    "inactive_package_flag",
]

# =========================
# Install / Supply-chain
# =========================
INSTALL_FEATURES_LIST = [
    "has_install_script",
    "num_install_scripts",
]

# =========================
# BUNI – Atomic behaviors (STATIC)
# =========================
BUNI_FEATURES_LIST = [
    "buni_DYNAMIC_CODE_EXEC",  # eval, new Function
    "buni_NETWORK_COMM",  # http, https, fetch
    "buni_PROCESS_EXECUTION",  # child_process
]

# 2. Combine into ALL_FEATURES (Final set)
ALL_FEATURES = (
        WEAKLINK_FEATURES_LIST +
        INSTALL_FEATURES_LIST +
        BUNI_FEATURES_LIST
)

# 3. Add the mandatory metadata columns
COLUMNS_TO_KEEP = ["package_name"] + ALL_FEATURES + ["label"]


def filter_dataset(input_csv_path, output_csv_path):
    # Load the dataset
    try:
        df = pd.read_csv(input_csv_path)

        # Filter columns: keep only those that exist in our list AND the dataframe
        existing_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]

        filtered_df = df[existing_columns]

        # Save the cleaned version
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"Success! Filtered dataset saved to: {output_csv_path}")
        print(f"Columns kept: {len(existing_columns)} out of {len(df.columns)} original columns.")
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found.")


# Example Usage:
if __name__ == "__main__":
    # Note: Using the same name for input and output will overwrite the original files.
    filter_dataset("npm_train.csv", "npm_train.csv")
    filter_dataset("npm_test.csv", "npm_test.csv")
    filter_dataset("npm_val_with_label.csv", "npm_val_with_label.csv")
    filter_dataset("npm_val_without_labels.csv", "npm_val_without_labels.csv")