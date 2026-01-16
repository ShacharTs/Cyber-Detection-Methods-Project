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
    "buni_DYNAMIC_CODE_EXEC",   # eval, new Function
    "buni_NETWORK_COMM",        # http, https, fetch
    "buni_PROCESS_EXECUTION",   # child_process
]



# =========================
# FINAL feature set
# =========================
ALL_FEATURES = (
    WEAKLINK_FEATURES_LIST +
    INSTALL_FEATURES_LIST +
    BUNI_FEATURES_LIST
)

INSTALL_KEYS = {"install", "preinstall", "postinstall", "prepare"}
