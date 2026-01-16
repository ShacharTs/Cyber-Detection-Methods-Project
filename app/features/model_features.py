# =========================
# Weak-Link
# =========================
WEAKLINK_FEATURES_LIST = [
    "num_dependencies",      # Valid: Part of the attack surface [cite: 127]
    "num_dev_dependencies",  # Valid: Part of the attack surface [cite: 127]
    # "description_length",  # REMOVE: Potential OF. Not a weak link signal
    # "num_js_files",        # REMOVE: Potential OF. Statistical noise, not a security risk
    "num_maintainers",       # Valid: W4 - Too many maintainers [cite: 243-248]
    "num_contributors",      # Valid: W5 - Too many contributors [cite: 262-264]
    # "repository_exists",   # REMOVE: Used only for data cleaning, not as a predictor
    # "license_exists",      # REMOVE: Used only for data cleaning, not as a predictor
    "inactive_package_flag", # Valid: W3 - Unmaintained package [cite: 218-219]
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
