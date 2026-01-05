"""
Supply-chain / Weak-Link feature extraction for npm packages.
Pure metadata + static signals only.
No execution, no network side effects.

Author: Shachar Tsrafati
"""

from datetime import datetime
from typing import Dict, Any

INSTALL_KEYS = {"install", "preinstall", "postinstall", "prepare"}


class WeakLinkFeatures:
    """
    Extracts Weak-Link (supply-chain risk) features
    from npm registry metadata + static package info.
    """

    # =========================================================
    # Public API
    # =========================================================

    @staticmethod
    def extract(
        registry_doc: Dict[str, Any],
        latest_pkg: Dict[str, Any],
        num_js_files: int = 0
    ) -> Dict[str, int | float]:
        """
        Main entry point.
        Returns a dict of Weak-Link features only.
        """

        features = {}

        features.update(
            WeakLinkFeatures._dependency_features(latest_pkg)
        )
        features.update(
            WeakLinkFeatures._ecosystem_features(registry_doc)
        )
        features.update(
            WeakLinkFeatures._metadata_features(latest_pkg)
        )
        features.update(
            WeakLinkFeatures._package_structure_features(
                latest_pkg, num_js_files
            )
        )
        features.update(
            WeakLinkFeatures._maintenance_features(
                registry_doc, latest_pkg
            )
        )
        features.update(
            WeakLinkFeatures._install_script_features(latest_pkg)
        )

        return features

    # =========================================================
    # Dependency / Ecosystem
    # =========================================================

    @staticmethod
    def _dependency_features(pkg: Dict[str, Any]) -> Dict[str, int]:
        deps = pkg.get("dependencies") or {}
        devdeps = pkg.get("devDependencies") or {}

        return {
            "num_dependencies": len(deps),
            "num_dev_dependencies": len(devdeps),
        }

    @staticmethod
    def _ecosystem_features(doc: Dict[str, Any]) -> Dict[str, int]:
        dependents = doc.get("dependents")

        return {
            "num_dependents": int(dependents) if dependents is not None else 0
        }

    # =========================================================
    # Metadata Quality
    # =========================================================

    @staticmethod
    def _metadata_features(pkg: Dict[str, Any]) -> Dict[str, int]:
        description = pkg.get("description") or ""

        return {
            "description_length": len(description),
            "repository_exists": 1 if pkg.get("repository") else 0,
            "license_exists": 1 if pkg.get("license") else 0,
        }

    # =========================================================
    # Package Structure
    # =========================================================

    @staticmethod
    def _package_structure_features(
        pkg: Dict[str, Any],
        num_js_files: int
    ) -> Dict[str, int]:
        return {
            "num_js_files": int(num_js_files),
            "has_bin": 1 if pkg.get("bin") else 0,
        }

    # =========================================================
    # Maintenance / Bus Factor
    # =========================================================

    @staticmethod
    def _maintenance_features(
        doc: Dict[str, Any],
        pkg: Dict[str, Any]
    ) -> Dict[str, int]:
        maintainers = doc.get("maintainers") or []
        contributors = pkg.get("contributors") or []

        last_update_days = WeakLinkFeatures._days_since_last_update(
            doc, pkg
        )

        return {
            "num_maintainers": len(maintainers),
            "num_contributors": len(contributors),
            "days_since_last_update": last_update_days,
            "inactive_package_flag": 1 if last_update_days > 365 else 0,
        }

    @staticmethod
    def _days_since_last_update(
        doc: Dict[str, Any],
        pkg: Dict[str, Any]
    ) -> int:
        time_info = doc.get("time") or {}

        latest_version = (
            (doc.get("dist-tags") or {}).get("latest")
        )

        ts = (
            time_info.get(latest_version)
            or time_info.get("modified")
        )

        if not ts:
            return 9999

        try:
            last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return (datetime.utcnow() - last).days
        except Exception:
            return 9999

    # =========================================================
    # Install-Time Attack Surface
    # =========================================================

    @staticmethod
    def _install_script_features(pkg: Dict[str, Any]) -> Dict[str, int]:
        scripts = pkg.get("scripts") or {}

        install_scripts = [
            k for k in scripts.keys() if k in INSTALL_KEYS
        ]

        complexity = sum(
            len(scripts[k]) for k in install_scripts
            if isinstance(scripts.get(k), str)
        )

        return {
            "has_install_script": 1 if install_scripts else 0,
            "num_install_scripts": len(install_scripts),
            "install_script_complexity": complexity,
        }
