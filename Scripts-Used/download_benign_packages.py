import requests
import os
import tarfile
import io
import time

OUTPUT_DIR = "benign_packages"
NUM_PACKAGES = 1000      # you can change this
PAGE_SIZE = 250          # maximum allowed by the API

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_top_packages(num=1000):
    """Fetch top npm packages sorted by popularity."""
    packages = []
    pages = num // PAGE_SIZE

    print(f"[+] Fetching metadata for top {num} npm packages...")

    for i in range(pages):
        url = f"https://api.npms.io/v2/search?q=not:deprecated&size={PAGE_SIZE}&from={i * PAGE_SIZE}"
        print(f"[+] Requesting page {i+1}/{pages}...")

        res = requests.get(url)
        res.raise_for_status()

        data = res.json()

        for pkg in data.get("results", []):
            name = pkg["package"]["name"]
            packages.append(name)

        time.sleep(0.1)  # avoid rate-limits

    print(f"[+] Retrieved {len(packages)} package names.")
    return packages


def download_and_extract(package_name):
    """Download and extract one npm package tarball."""
    try:
        # Get metadata to find tarball URL
        meta_url = f"https://registry.npmjs.org/{package_name}"
        meta_res = requests.get(meta_url, timeout=10)
        meta_res.raise_for_status()
        
        meta = meta_res.json()
        version = meta["dist-tags"]["latest"]
        tarball_url = meta["versions"][version]["dist"]["tarball"]

        print(f"[.] Downloading {package_name}@{version}")

        tarball_res = requests.get(tarball_url, timeout=15)
        tarball_res.raise_for_status()

        pkg_dir = os.path.join(OUTPUT_DIR, package_name.replace("/", "_"))
        os.makedirs(pkg_dir, exist_ok=True)

        # Extract tar.gz from memory
        with tarfile.open(fileobj=io.BytesIO(tarball_res.content), mode="r:gz") as tar:
            tar.extractall(pkg_dir)

        print(f"[+] Extracted {package_name} → {pkg_dir}")

    except Exception as e:
        print(f"[!] Failed to download {package_name}: {e}")


def main():
    packages = fetch_top_packages(NUM_PACKAGES)

    print("[+] Starting download and extraction...\n")

    for i, pkg in enumerate(packages):
        print(f"[{i+1}/{len(packages)}]")
        download_and_extract(pkg)

    print("\n[+] All done!")
    print(f"[+] Benign packages saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
