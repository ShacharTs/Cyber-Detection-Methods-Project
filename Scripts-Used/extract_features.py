#!/usr/bin/env python3
"""
extract_features.py

Build features.csv from two directories:
- malicious_samples/  (label=1)
- benign_packages/    (label=0)

Usage:
  python extract_features.py --malicious malicious_samples --benign benign_packages --out features.csv --verbose
"""

import argparse
import json
import os
import re
import math
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# DONAPI Behavior Mapping
# ----------------------------
API_TO_ATOMIC = {
    "fs.": "FILE_OP",
    "fsPromises.": "FILE_OP",

    "http.": "NET_OP",
    "https.": "NET_OP",
    "request(": "NET_OP",
    "fetch(": "NET_OP",
    "axios(": "NET_OP",
    "net.": "NET_OP",
    "dns.": "NET_OP",

    "child_process.exec": "PROC_EXEC",
    "child_process.execSync": "PROC_EXEC",
    "child_process.spawn": "PROC_EXEC",
    "child_process.spawnSync": "PROC_EXEC",

    "eval(": "DYN_EXEC",
    "Function(": "DYN_EXEC",

    "process.env": "ENV_ACCESS",
    "crypto.": "CRYPTO_OP",
    "process.chdir": "FS_CONTEXT",
}

ATOMIC_TO_BEHAVIOR = {
    "FILE_OP": "SENSITIVE_FILE_OP",
    "NET_OP": "NETWORK_COMM",
    "PROC_EXEC": "PROCESS_EXECUTION",
    "DYN_EXEC": "DYNAMIC_CODE_EXEC",
    "ENV_ACCESS": "ENVIRONMENT_ACCESS",
    "CRYPTO_OP": "CRYPTO_USAGE",
    "FS_CONTEXT": "FILESYSTEM_CONTEXT",
}

# ----------------------------
# Utilities
# ----------------------------
JS_EXTS = (".js", ".mjs", ".cjs")
API_TOKEN_ORDER = [
    "fs.", "fsPromises.", "http.", "https.", "request(", "fetch(", "axios(",
    "child_process.exec", "child_process.execSync", "child_process.spawn",
    "child_process.spawnSync", "eval(", "Function(", "process.env", "process.chdir",
    "crypto.", "net.", "dns."
]

RE_SPECIAL_CHARS = re.compile(r'[%\$\\@{};]')
RE_STRING_LITERAL = re.compile(r"(?:'([^']*)'|\"([^\"]*)\"|`([^`]*)`)", re.S)
RE_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
RE_CHILDPROC = re.compile(
    r"(?:child_process\.(exec|execSync|spawn|spawnSync)\s*\(\s*['\"]([^'\"]+)['\"])",
    re.I
)
RE_FS_OPS = re.compile(
    r"\bfs\.(writeFile|writeFileSync|readFile|readFileSync|chmod|unlink|rm|appendFile)\b", re.I
)
RE_NET = re.compile(r"\b(?:https?://|fetch\(|axios\(|request\(|https?\.get|http\.request)\b", re.I)
RE_DYNAMIC = re.compile(r"\beval\b|\bnew Function\b", re.I)
RE_ENV = re.compile(r"\bprocess\.env\b", re.I)

MAX_FILE_SIZE = 5_000_000  # skip giant files


def safe_read(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    e = 0.0
    L = len(s)
    for v in freq.values():
        p = v / L
        e -= p * math.log2(p)
    return e


# ----------------------------
# Feature extraction
# ----------------------------
def list_js_files(package_root: Path) -> List[Path]:
    js_files = []
    for root, _, files in os.walk(package_root):
        for fname in files:
            if fname.endswith(JS_EXTS):
                js_files.append(Path(root) / fname)
    return js_files


def extract_obfuscation_features_from_text(code: str) -> Dict[str, Any]:
    total_len = len(code)
    if total_len == 0:
        return {
            "line_ratio": 0.0, "space_ratio": 0.0, "num_long_strings": 0,
            "special_char_count": 0, "substring_calls": 0,
            "identifier_entropy": 0.0, "avg_identifier_length": 0.0,
            "num_lines": 0, "max_string_len": 0
        }

    compressed = code.replace("\n", "").replace("\r", "")
    line_ratio = len(compressed) / max(1, total_len)
    space_ratio = code.count(" ") / max(1, total_len)

    strings = [(m.group(1) or m.group(2) or m.group(3) or "") for m in RE_STRING_LITERAL.finditer(code)]
    num_long_strings = sum(1 for s in strings if len(s) > 50)
    max_string_len = max((len(s) for s in strings), default=0)

    special_char_count = len(RE_SPECIAL_CHARS.findall(code))
    substring_calls = len(re.findall(r"\b(?:substring|substr|charAt|slice)\s*\(", code))

    identifiers = RE_IDENTIFIER.findall(code)
    identifier_entropy = shannon_entropy("".join(identifiers)) if identifiers else 0.0
    avg_identifier_length = sum(len(i) for i in identifiers) / max(1, len(identifiers))
    num_lines = code.count("\n") + 1

    return {
        "line_ratio": float(line_ratio),
        "space_ratio": float(space_ratio),
        "num_long_strings": int(num_long_strings),
        "max_string_len": int(max_string_len),
        "special_char_count": int(special_char_count),
        "substring_calls": int(substring_calls),
        "identifier_entropy": float(identifier_entropy),
        "avg_identifier_length": float(avg_identifier_length),
        "num_lines": int(num_lines),
    }


def extract_behavior_counts_and_api_sequence(code: str) -> Tuple[Dict[str, int], List[str]]:
    counts = {
        "net_requests": len(RE_NET.findall(code)) if RE_NET.search(code) else 0,
        "fs_ops": len(RE_FS_OPS.findall(code)) if RE_FS_OPS.search(code) else 0,
        "proc_exec": len(RE_CHILDPROC.findall(code)) if RE_CHILDPROC.search(code) else 0,
        "dynamic_exec": len(RE_DYNAMIC.findall(code)) if RE_DYNAMIC.search(code) else 0,
        "env_access": len(RE_ENV.findall(code)) if RE_ENV.search(code) else 0,
    }

    positions = []
    for token in API_TOKEN_ORDER:
        for m in re.finditer(re.escape(token), code, re.I):
            positions.append((m.start(), token))
    positions.sort()
    api_seq = [tok for _, tok in positions]
    return counts, api_seq


def map_api_sequence_to_behaviors(api_seq: List[str]) -> List[str]:
    behavior_seq = []
    for token in api_seq:
        atomic = None
        for api_prefix, atomic_op in API_TO_ATOMIC.items():
            if token.startswith(api_prefix):
                atomic = atomic_op
                break
        if atomic is None:
            continue
        behavior = ATOMIC_TO_BEHAVIOR.get(atomic)
        if behavior:
            behavior_seq.append(behavior)
    return behavior_seq


def ngram_counts(seq: List[str], n: int) -> Counter:
    if not seq or len(seq) < n:
        return Counter()
    grams = Counter()
    for i in range(len(seq) - n + 1):
        grams["|".join(seq[i:i + n])] += 1
    return grams


# ----------------------------
# Package processing
# ----------------------------
def discover_package_roots(root_dir: str) -> List[Path]:
    roots = []
    for entry in os.listdir(root_dir):
        p = Path(root_dir) / entry
        if not p.exists():
            continue
        found = False
        for root, _, files in os.walk(p):
            if "package.json" in files:
                roots.append(p)
                found = True
                break
        if not found and (p / "package.json").exists():
            roots.append(p)
    return roots


def process_package_dir(pkg_path: Path, label: str) -> Dict[str, Any]:
    pkgjson = None
    code_root = None

    for root, _, files in os.walk(pkg_path):
        if "package.json" in files:
            pkgjson = Path(root) / "package.json"
            code_root = Path(root)
            break

    if not pkgjson:
        return {}

    try:
        pj = json.loads(pkgjson.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pj = {}

    # metadata
    num_deps = len(pj.get("dependencies", {})) if isinstance(pj.get("dependencies", {}), dict) else 0
    num_maintainers = len(pj.get("maintainers", [])) if isinstance(pj.get("maintainers", []), list) else 0
    has_scripts = int("scripts" in pj and bool(pj.get("scripts")))
    repo_exists = int(bool(pj.get("repository")))
    desc_len = len(pj.get("description", "") or "")

    js_files = list_js_files(code_root)

    file_count = 0
    seq_all: List[str] = []
    behavior_counts = Counter()
    obf_agg = {
        "line_ratio": [], "space_ratio": [], "num_long_strings": [], "max_string_len": [],
        "special_char_count": [], "substring_calls": [], "identifier_entropy": [],
        "avg_identifier_length": [], "num_lines": []
    }

    for f in js_files:
        txt = safe_read(f)
        if not txt:
            continue
        file_count += 1

        # obfuscation (per file)
        obf = extract_obfuscation_features_from_text(txt)
        for k in obf_agg:
            obf_agg[k].append(obf[k])

        # behaviors + sequence
        bc, api_seq = extract_behavior_counts_and_api_sequence(txt)
        for k, v in bc.items():  # IMPORTANT: accumulate bf_*
            behavior_counts[k] += v

        behavior_seq = map_api_sequence_to_behaviors(api_seq)
        seq_all.extend(behavior_seq)

    # aggregate obfuscation
    obf_features = {}
    for k, lst in obf_agg.items():
        if not lst:
            obf_features[k] = 0.0
        else:
            if k in ("max_string_len", "special_char_count", "num_long_strings", "num_lines"):
                obf_features[k] = float(max(lst))
            else:
                obf_features[k] = float(sum(lst) / len(lst))

    # ngrams over behavior sequence
    uni = ngram_counts(seq_all, 1)
    bi = ngram_counts(seq_all, 2)
    tri = ngram_counts(seq_all, 3)

    row = {
        "path": str(pkg_path),
        "label": str(label),
        "num_js_files": int(file_count),
        "num_dependencies": int(num_deps),
        "num_maintainers": int(num_maintainers),
        "has_scripts": int(has_scripts),
        "repository_exists": int(repo_exists),
        "description_length": int(desc_len),
    }

    row.update({f"bf_{k}": int(behavior_counts.get(k, 0))
                for k in ["net_requests", "fs_ops", "proc_exec", "dynamic_exec", "env_access"]})

    row.update(obf_features)

    row["_uni"] = dict(uni)
    row["_bi"] = dict(bi)
    row["_tri"] = dict(tri)

    return row


def build_dataset(malicious_dir: str, benign_dir: str, verbose=True) -> pd.DataFrame:
    rows = []

    if os.path.isdir(malicious_dir):
        mal_roots = discover_package_roots(malicious_dir)
        if verbose:
            print(f"[+] Found {len(mal_roots)} malicious package roots")
        for r in tqdm(mal_roots, desc="malicious"):
            rec = process_package_dir(r, label="malicious")
            if rec:
                rows.append(rec)
    else:
        print("[!] malicious_dir not found:", malicious_dir)

    if os.path.isdir(benign_dir):
        ben_roots = discover_package_roots(benign_dir)
        if verbose:
            print(f"[+] Found {len(ben_roots)} benign package roots")
        for r in tqdm(ben_roots, desc="benign"):
            rec = process_package_dir(r, label="benign")
            if rec:
                rows.append(rec)
    else:
        print("[!] benign_dir not found:", benign_dir)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # global vocab for ngrams
    uni_all, bi_all, tri_all = Counter(), Counter(), Counter()
    for d in df[["_uni", "_bi", "_tri"]].to_dict(orient="records"):
        uni_all.update(d["_uni"])
        bi_all.update(d["_bi"])
        tri_all.update(d["_tri"])

    TOP_U = [g for g, _ in uni_all.most_common(50)]
    TOP_B = [g for g, _ in bi_all.most_common(50)]
    TOP_T = [g for g, _ in tri_all.most_common(30)]

    for g in TOP_U:
        df[f"buni_{g}"] = df["_uni"].apply(lambda d: d.get(g, 0) if isinstance(d, dict) else 0)
    for g in TOP_B:
        df[f"bbi_{g}"] = df["_bi"].apply(lambda d: d.get(g, 0) if isinstance(d, dict) else 0)
    for g in TOP_T:
        df[f"btri_{g}"] = df["_tri"].apply(lambda d: d.get(g, 0) if isinstance(d, dict) else 0)

    df = df.drop(columns=["_uni", "_bi", "_tri"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicious", required=True)
    ap.add_argument("--benign", required=True)
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = build_dataset(args.malicious, args.benign, verbose=args.verbose)
    if df.empty:
        print("[!] No data extracted.")
        return

    df.to_csv(args.out, index=False)
    print(f"[+] Wrote {args.out} with shape {df.shape}")


if __name__ == "__main__":
    main()

