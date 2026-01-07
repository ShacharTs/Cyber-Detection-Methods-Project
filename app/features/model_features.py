# =========================
# Weak-Link
# =========================
WEAKLINK_FEATURES_LIST = [
    "num_dependencies",
    "num_dev_dependencies",
    "description_length",
    "num_js_files",
    "num_maintainers",
    "num_contributors",
    "repository_exists",
    "license_exists",
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
# DONAPI
# =========================
DONAPI_FEATURES_LIST = [
    "bf_net_requests",
    "bf_fs_ops",
    "bf_proc_exec",
    "bf_dynamic_exec",
    "bf_env_access",
]

# =========================
# BUNI – Atomic behaviors
# =========================
BUNI_FEATURES_LIST = [
    "buni_DYNAMIC_CODE_EXEC",
    "buni_NETWORK_COMM",
    "buni_PROCESS_EXECUTION",
]

# =========================
# Static / Obfuscation
# =========================
STATIC_FEATURES_LIST = [
    "identifier_entropy",
    "num_lines",
]

# =========================
# FINAL feature set
# =========================
ALL_FEATURES = (
    WEAKLINK_FEATURES_LIST +
    INSTALL_FEATURES_LIST +
    DONAPI_FEATURES_LIST +
    BUNI_FEATURES_LIST +
    STATIC_FEATURES_LIST
)

INSTALL_KEYS = {"install", "preinstall", "postinstall", "prepare"}
