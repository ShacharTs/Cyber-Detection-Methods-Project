import pandas as pd

# 1. Define the Feature Lists (as provided)
WEAKLINK_FEATURES_LIST = [
    "num_dependencies",
    "num_dev_dependencies",
    "num_maintainers",
    "num_contributors",
    "inactive_package_flag",
]

INSTALL_FEATURES_LIST = [
    "has_install_script",
    "num_install_scripts",
]

DONAPI_FEATURES_LIST = [
    "bf_net_requests",
    "bf_fs_ops",
    "bf_proc_exec",
    "bf_dynamic_exec",
    "bf_env_access",
]

BUNI_FEATURES_LIST = [
    "buni_DYNAMIC_CODE_EXEC",
    "buni_NETWORK_COMM",
    "buni_PROCESS_EXECUTION",
]

STATIC_FEATURES_LIST = [
    "identifier_entropy",
    "num_lines",
]

# 2. Combine into ALL_FEATURES
ALL_FEATURES = (
        WEAKLINK_FEATURES_LIST +
        INSTALL_FEATURES_LIST +
        DONAPI_FEATURES_LIST +
        BUNI_FEATURES_LIST +
        STATIC_FEATURES_LIST
)

# 3. Add the mandatory metadata columns
# We use a list to ensure order: package_name first, then features, then label
COLUMNS_TO_KEEP = ["package_name"] + ALL_FEATURES + ["label"]


def filter_dataset(input_csv_path, output_csv_path):
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Filter columns: keep only those that exist in our list AND the dataframe
    # This prevents errors if a column is missing from the CSV
    existing_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]

    filtered_df = df[existing_columns]

    # Save the cleaned version
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Success! Filtered dataset saved to: {output_csv_path}")
    print(f"Columns kept: {len(existing_columns)} out of {len(df.columns)} original columns.")

# Example Usage:
filter_dataset("npm_train.csv", "npm_train.csv")
filter_dataset("npm_test.csv", "npm_test.csv")
filter_dataset("npm_val_with_label.csv", "npm_val_with_label.csv")
filter_dataset("npm_val_without_labels.csv", "npm_val_without_labels.csv")