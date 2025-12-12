#!/usr/bin/env python3
"""
DONAPI-Lite: static, safe approximation of DONAPI pipeline.

- Scans two directories: malicious_samples/ and benign_packages/
- Extracts metadata, obfuscation, behavior and sequence n-gram features
- Trains a RandomForest classifier and evaluates it
- Saves features.csv and rf_model.joblib

Usage:
    python donapi_lite.py --malicious malicious_samples --benign benign_packages --out features.csv --model rf_model.joblib
"""

import argparse
import json
import os
import re
import math
import io
import tarfile
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ----------------------------
# Utilities
# ----------------------------
JS_EXTS = (".js", ".mjs", ".cjs")
API_TOKEN_ORDER = [
    # tokens we want to recognize in sequence order (sensible reduced set)
    "fs.", "fsPromises.", "http.", "https.", "request(", "fetch(", "axios(", 
    "child_process.exec", "child_process.execSync", "child_process.spawn",
    "child_process.spawnSync", "eval(", "Function(", "process.env", "process.chdir",
    "crypto.", "net.", "dns."
]

RE_SPECIAL_CHARS = re.compile(r'[%\$\\@{};]')
RE_STRING_LITERAL = re.compile(r"(?:'([^']*)'|\"([^\"]*)\"|`([^`]*)`)", re.S)
RE_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
RE_CHILDPROC = re.compile(r"(?:child_process\.(exec|execSync|spawn|spawnSync)\s*\(\s*['\"]([^'\"]+)['\"])",
                          re.I)
RE_FS_OPS = re.compile(r"\bfs\.(writeFile|writeFileSync|readFile|readFileSync|chmod|unlink|rm|appendFile)\b", re.I)
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
def extract_entry_points(pkg_json: Dict[str, Any]) -> List[str]:
    entries = []
    if not isinstance(pkg_json, dict):
        return entries
    for f in ("main", "exports", "imports", "bin"):
        val = pkg_json.get(f)
        if isinstance(val, str):
            entries.append(val)
        elif isinstance(val, dict):
            for v in val.values():
                if isinstance(v, str):
                    entries.append(v)
    scripts = pkg_json.get("scripts", {})
    if isinstance(scripts, dict):
        for v in scripts.values():
            if isinstance(v, str):
                entries.append(v)
    return entries

def list_js_files(package_root: Path) -> List[Path]:
    js_files = []
    for root, dirs, files in os.walk(package_root):
        for fname in files:
            if fname.endswith(JS_EXTS):
                p = Path(root) / fname
                js_files.append(p)
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
    strings = [ (m.group(1) or m.group(2) or m.group(3) or "") for m in RE_STRING_LITERAL.finditer(code) ]
    num_long_strings = sum(1 for s in strings if len(s) > 50)
    max_string_len = max((len(s) for s in strings), default=0)
    special_char_count = len(RE_SPECIAL_CHARS.findall(code))
    substring_calls = len(re.findall(r"\b(?:substring|substr|charAt|slice)\s*\(", code))
    identifiers = RE_IDENTIFIER.findall(code)
    identifier_entropy = shannon_entropy("".join(identifiers)) if identifiers else 0.0
    avg_identifier_length = sum(len(idf) for idf in identifiers) / max(1, len(identifiers))
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
        "num_lines": int(num_lines)
    }

def extract_behavior_counts_and_sequence(code: str) -> Tuple[Dict[str,int], List[str]]:
    """
    Return counts of key behaviors and a sequence (ordered tokens) of API hits.
    """
    counts = {
        "net_requests": int(bool(RE_NET.search(code))) and len(RE_NET.findall(code)) or 0,
        "fs_ops": int(bool(RE_FS_OPS.search(code))) and len(RE_FS_OPS.findall(code)) or 0,
        "proc_exec": int(bool(RE_CHILDPROC.search(code))) and len(RE_CHILDPROC.findall(code)) or 0,
        "dynamic_exec": int(bool(RE_DYNAMIC.search(code))) and len(RE_DYNAMIC.findall(code)) or 0,
        "env_access": int(bool(RE_ENV.search(code))) and len(RE_ENV.findall(code)) or 0
    }
    # Build ordered token sequence by scanning in-file left->right
    seq = []
    lowered = code  # keep raw for matching tokens defined above (case-insensitive)
    # For speed, scan for API tokens positions
    positions = []
    for token in API_TOKEN_ORDER:
        for m in re.finditer(re.escape(token), lowered, re.I):
            positions.append((m.start(), token))
    positions.sort()
    seq = [tok for _, tok in positions]
    return counts, seq

def ngram_counts(seq: List[str], n: int) -> Counter:
    if not seq or len(seq) < n:
        return Counter()
    grams = Counter()
    for i in range(len(seq) - n + 1):
        grams["|".join(seq[i:i+n])] += 1
    return grams

# ----------------------------
# Package processor
# ----------------------------
def process_package_dir(pkg_path: Path, label: int) -> Dict[str, Any]:
    """
    pkg_path: path that contains package/ or package.json (we find package.json automatically)
    label: 1 malicious, 0 benign
    """
    # Try to find package.json within path or its subfolders
    pkgjson = None
    for root, dirs, files in os.walk(pkg_path):
        if "package.json" in files:
            pkgjson = Path(root) / "package.json"
            code_root = Path(root)  # assume JS under same folder
            break
    if not pkgjson:
        # fallback: if path itself is a package.json file
        if pkg_path.is_file() and pkg_path.name == "package.json":
            pkgjson = pkg_path
            code_root = pkg_path.parent
        else:
            # nothing found
            return {}

    try:
        pj = json.loads(pkgjson.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pj = {}

    # metadata features
    num_deps = len(pj.get("dependencies", {})) if isinstance(pj.get("dependencies", {}), dict) else 0
    num_maintainers = len(pj.get("maintainers", [])) if isinstance(pj.get("maintainers", []), list) else 0
    has_scripts = int("scripts" in pj and bool(pj.get("scripts")))
    repo_exists = int(bool(pj.get("repository")))
    desc_len = len(pj.get("description","") or "")

    # collect JS content
    js_files = list_js_files(code_root)
    combined = []
    file_count = 0
    seq_all = []
    behavior_counts = Counter()
    obf_agg = {
        "line_ratio": [], "space_ratio": [], "num_long_strings": [], "max_string_len": [],
        "special_char_count": [], "substring_calls": [], "identifier_entropy": [], "avg_identifier_length": [], "num_lines": []
    }

    for f in js_files:
        txt = safe_read(f)
        if not txt:
            continue
        file_count += 1
        # obfuscation features per file
        obf = extract_obfuscation_features_from_text(txt)
        for k in obf_agg:
            obf_agg[k].append(obf[k])
        # behavior counts & sequence
        bc, seq = extract_behavior_counts_and_sequence(txt)
        for k,v in bc.items():
            behavior_counts[k] += v
        seq_all.extend(seq)

    # aggregate obfuscation features by mean / max
    obf_features = {}
    for k, lst in obf_agg.items():
        if not lst:
            obf_features[k] = 0.0
        else:
            # for some features use mean, for string length use max
            if k in ("max_string_len", "special_char_count", "num_long_strings", "num_lines"):
                obf_features[k] = float(max(lst))
            else:
                obf_features[k] = float(sum(lst) / len(lst))

    # n-gram extraction (uni/bi/tri)
    uni = ngram_counts(seq_all, 1)
    bi = ngram_counts(seq_all, 2)
    tri = ngram_counts(seq_all, 3)

    # top-k n-grams (we'll choose a few global later)
    row = {
        "path": str(pkg_path),
        "label": int(label),
        "num_js_files": int(file_count),
        "num_dependencies": int(num_deps),
        "num_maintainers": int(num_maintainers),
        "has_scripts": int(has_scripts),
        "repository_exists": int(repo_exists),
        "description_length": int(desc_len),
    }
    # behavior counts
    row.update({f"bf_{k}": int(behavior_counts.get(k,0)) for k in ["net_requests","fs_ops","proc_exec","dynamic_exec","env_access"]})
    # obfuscation
    row.update(obf_features)
    # attach top ngrams counts placeholders (we will vectorize globally)
    row["_uni"] = dict(uni)
    row["_bi"] = dict(bi)
    row["_tri"] = dict(tri)
    return row

# ----------------------------
# Build dataset
# ----------------------------
def discover_package_roots(root_dir: str) -> List[Path]:
    """
    Return list of candidate package roots to process. We consider:
      - directories that contain 'package/' folder
      - directories that contain a package.json somewhere inside
    """
    roots = []
    for entry in os.listdir(root_dir):
        p = Path(root_dir) / entry
        if not p.exists():
            continue
        # if tar extracted folder pattern like name-version/package/...
        # try to find any package.json under p
        found = False
        for root, dirs, files in os.walk(p):
            if "package.json" in files:
                roots.append(p)
                found = True
                break
        if not found and (p / "package.json").exists():
            roots.append(p)
    return roots

def build_full_dataset(malicious_dir: str, benign_dir: str, verbose=True) -> pd.DataFrame:
    rows = []
    # malicious
    if os.path.isdir(malicious_dir):
        mal_roots = discover_package_roots(malicious_dir)
        if verbose:
            print(f"[+] Found {len(mal_roots)} malicious package roots")
        for r in tqdm(mal_roots, desc="malicious"):
            rec = process_package_dir(r, label=1)
            if rec:
                rows.append(rec)
    else:
        print("[!] malicious_dir not found:", malicious_dir)
    # benign
    if os.path.isdir(benign_dir):
        ben_roots = discover_package_roots(benign_dir)
        if verbose:
            print(f"[+] Found {len(ben_roots)} benign package roots")
        for r in tqdm(ben_roots, desc="benign"):
            rec = process_package_dir(r, label=0)
            if rec:
                rows.append(rec)
    else:
        print("[!] benign_dir not found:", benign_dir)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Build global ngram vocabulary (choose top K uni/bi/tri across dataset)
    uni_all = Counter()
    bi_all = Counter()
    tri_all = Counter()
    for d in df[["_uni","_bi","_tri"]].to_dict(orient="records"):
        uni_all.update(d["_uni"])
        bi_all.update(d["_bi"])
        tri_all.update(d["_tri"])
    TOP_U = [g for g,_ in uni_all.most_common(50)]
    TOP_B = [g for g,_ in bi_all.most_common(50)]
    TOP_T = [g for g,_ in tri_all.most_common(30)]
    # expand dataframe with ngram features
    for g in TOP_U:
        df[f"uni_{g}"] = df["_uni"].apply(lambda d: d.get(g,0) if isinstance(d, dict) else 0)
    for g in TOP_B:
        df[f"bi_{g}"] = df["_bi"].apply(lambda d: d.get(g,0) if isinstance(d, dict) else 0)
    for g in TOP_T:
        df[f"tri_{g}"] = df["_tri"].apply(lambda d: d.get(g,0) if isinstance(d, dict) else 0)
    # drop raw ngram columns
    df = df.drop(columns=["_uni","_bi","_tri"])
    return df

# ----------------------------
# Train & evaluate
# ----------------------------
def train_and_evaluate(df: pd.DataFrame, model_out: str, csv_out: str):
    if df.empty:
        print("[!] Empty dataset.")
        return
    # prepare features
    drop_cols = ["path","label"]
    X = df[[c for c in df.columns if c not in drop_cols]].fillna(0).astype(float)
    y = df["label"].astype(int)
    # ensure at least two classes
    if len(np.unique(y)) < 2:
        print("[!] Need at least two classes to train. Found classes:", np.unique(y))
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("[+] Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    if hasattr(clf, "predict_proba") and len(np.unique(y_test)) > 1:
        y_proba = clf.predict_proba(X_test)[:,1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"[+] ROC-AUC: {auc:.4f}")
        except Exception:
            pass
    # feature importances
    feats = X.columns.tolist()
    imps = sorted(zip(feats, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:30]
    print("[+] Top features:")
    for n, imp in imps:
        print(f"    {n}: {imp:.4f}")
    # save
    joblib.dump(clf, model_out)
    df.to_csv(csv_out, index=False)
    print(f"[+] Saved model -> {model_out}, features -> {csv_out}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malicious", required=True, help="Directory of malicious samples")
    ap.add_argument("--benign", required=True, help="Directory of benign packages")
    ap.add_argument("--out-csv", default="features_donapi_lite.csv")
    ap.add_argument("--out-model", default="rf_donapi_lite.joblib")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    df = build_full_dataset(args.malicious, args.benign, verbose=args.verbose)
    if df.empty:
        print("[!] No data extracted.")
        return
    print(f"[+] Extracted dataset with shape: {df.shape}")
    train_and_evaluate(df, model_out=args.out_model, csv_out=args.out_csv)

if __name__ == "__main__":
    main()
