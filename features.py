FEATURES = [
    # =========================
    # Identifier
    # =========================
    "package_name",              # Identifier (not used for modeling)

    # =========================
    # Weak-Link / Supply-Chain
    # (Metadata, ecosystem, maintenance)
    # =========================
    "num_dependencies",           # Weak-Link: dependency complexity
    "num_dev_dependencies",       # Weak-Link: dev dependency surface
    "num_dependents",             # Weak-Link: ecosystem trust / popularity

    "description_length",         # Weak-Link: metadata quality
    "repository_exists",          # Weak-Link: transparency
    "license_exists",             # Weak-Link: legitimacy

    "num_js_files",               # Weak-Link: package size / complexity
    "has_bin",                    # Weak-Link: executable exposure

    "num_maintainers",            # Weak-Link: bus factor
    "num_contributors",           # Weak-Link: community involvement

    "days_since_last_update",     # Weak-Link: maintenance freshness
    "inactive_package_flag",      # Weak-Link: abandoned package signal

    "has_install_script",         # Weak-Link: install-time attack surface
    "num_install_scripts",        # Weak-Link: install script intensity
    "install_script_complexity",  # Weak-Link: script complexity proxy

    # =========================
    # DONPAI – Behavioral Intent
    # (static behavioral signals, no execution)
    # =========================
    "bf_net_requests",            # DONPAI: network behavior intent
    "bf_fs_ops",                  # DONPAI: filesystem behavior intent
    "bf_proc_exec",               # DONPAI: process execution intent
    "bf_dynamic_exec",            # DONPAI: dynamic code execution intent
    "bf_env_access",              # DONPAI: environment access intent

    # =========================
    # DONPAI Too
    # BUNI – Malicious Capabilities
    # (specific malicious primitives)
    # =========================
    "buni_DYNAMIC_CODE_EXEC",     # BUNI: eval / Function usage
    "buni_SENSITIVE_FILE_OP",     # BUNI: sensitive file operations
    "buni_NETWORK_COMM",          # BUNI: network communication
    "buni_ENVIRONMENT_ACCESS",    # BUNI: process.env access
    "buni_CRYPTO_USAGE",          # BUNI: crypto API usage
    "buni_FILESYSTEM_CONTEXT",    # BUNI: fs module usage
    "buni_PROCESS_EXECUTION",     # BUNI: child_process execution

    # =========================
    # DONPAI Too
    # Static Code Analysis
    # (structure, obfuscation, style)
    # =========================
    "identifier_entropy",         # Static: identifier randomness / obfuscation
    "avg_identifier_length",      # Static: identifier style
    "special_char_count",         # Static: obfuscation indicator
    "num_long_strings",           # Static: payload / encoded data indicator
    "max_string_len",             # Static: longest string length
    "num_lines",                  # Static: code size
    "line_ratio",                 # Static: density / packing

    # =========================
    # Label
    # =========================
    "label"                       # Target: 1 = malware, 0 = benign
]

## Shachar's side of the features
WEAKLINK_FEATURES_LIST = [
    "num_dependencies",           # Dependency complexity
    "num_dev_dependencies",       # Dev dependency surface
    "num_dependents",             # Ecosystem trust / popularity

    "description_length",         # Metadata quality
    "repository_exists",          # Transparency
    "license_exists",             # Legitimacy

    "num_js_files",               # Package size / complexity
    "has_bin",                    # Executable exposure

    "num_maintainers",            # Bus factor
    "num_contributors",           # Community involvement

    "days_since_last_update",     # Maintenance freshness
    "inactive_package_flag",      # Abandonment signal

    "has_install_script",         # Install-time attack surface
    "num_install_scripts",        # Install script intensity
    "install_script_complexity"   # Script complexity proxy
]




INSTALL_KEYS = {"install", "preinstall", "postinstall", "prepare"}
