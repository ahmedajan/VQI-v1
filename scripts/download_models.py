"""Download pre-trained VQI models from GitHub Releases.

Usage:
    python scripts/download_models.py

Downloads vqi_rf_model.joblib and vqi_v_rf_model.joblib to the models/ directory.
"""

import os
import sys
import urllib.request

# Update these URLs after creating the GitHub Release
RELEASE_BASE = "https://github.com/YOUR_USERNAME/VQI/releases/download/v1.0"

MODELS = {
    "vqi_rf_model.joblib": f"{RELEASE_BASE}/vqi_rf_model.joblib",
    "vqi_v_rf_model.joblib": f"{RELEASE_BASE}/vqi_v_rf_model.joblib",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def download_file(url, dest):
    """Download a file with progress indicator."""
    print(f"Downloading {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print(f"\n  Saved to {dest}")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"  Please download manually from: {url}")
        return False
    return True


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
        sys.stdout.flush()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_ok = True
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"{filename} already exists, skipping.")
            continue
        if not download_file(url, dest):
            all_ok = False

    if all_ok:
        print("\nAll models downloaded successfully.")
    else:
        print("\nSome downloads failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
