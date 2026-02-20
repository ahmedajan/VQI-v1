"""
Dataset Inventory Script (Sub-task 1.2)

Walks each dataset directory in D:/VQI/Datasets/, counts speakers,
utterances, samples audio properties (sample rate, channels, bit depth),
estimates total duration, and outputs a summary CSV.

Usage:
    python implementation/scripts/inventory_datasets.py

Output:
    blueprint/dataset_inventory.csv
"""

import os
import sys
import csv
import random
import time
from pathlib import Path
from collections import defaultdict

import json
import subprocess

import soundfile as sf
import torchaudio

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

DATASETS_ROOT = Path("D:/VQI/Datasets")
OUTPUT_CSV = Path("D:/VQI/blueprint/dataset_inventory.csv")

# Number of files to sample for duration / format checks per dataset
SAMPLE_SIZE = 200

AUDIO_EXTENSIONS = {".wav", ".flac", ".m4a", ".aac", ".ogg", ".mp3", ".opus"}


# ──────────────────────────────────────────────────────────────
# Dataset-specific discovery functions
# ──────────────────────────────────────────────────────────────

def discover_voxceleb1(root: Path):
    """
    VoxCeleb1 structure: wav/{speaker_id}/{video_id}/{NNNNN}.wav
    Meta file provides dev/test split info.
    """
    wav_dir = root / "wav"
    if not wav_dir.exists():
        return None

    meta_path = root / "vox1_meta.csv"
    dev_speakers = set()
    test_speakers = set()

    if meta_path.exists():
        with open(meta_path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 5:
                    spk_id = row[0].strip()
                    split = row[4].strip()
                    if split == "dev":
                        dev_speakers.add(spk_id)
                    elif split == "test":
                        test_speakers.add(spk_id)

    # Walk the wav directory
    speakers = {}
    all_files = []
    for spk_dir in sorted(wav_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        spk_files = []
        for video_dir in spk_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for f in video_dir.iterdir():
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    spk_files.append(f)
        speakers[spk_id] = len(spk_files)
        all_files.extend(spk_files)

    # Separate into dev and test based on meta
    dev_count = sum(v for k, v in speakers.items() if k in dev_speakers)
    test_count = sum(v for k, v in speakers.items() if k in test_speakers)
    dev_spk_count = sum(1 for k in speakers if k in dev_speakers)
    test_spk_count = sum(1 for k in speakers if k in test_speakers)

    return [{
        "dataset": "VoxCeleb1",
        "speakers": len(speakers),
        "utterances": len(all_files),
        "files": all_files,
        "note": f"dev={dev_spk_count} spk/{dev_count} utt, test={test_spk_count} spk/{test_count} utt",
    }]


def discover_voxceleb2(root: Path):
    """
    VoxCeleb2 structure: aac/{speaker_id}/{video_id}/{NNNNN}.m4a (or wav)
    May also be in dev/aac/ or test/aac/ subdirectories.
    """
    meta_path = root / "vox2_meta.csv"
    dev_speakers = set()
    test_speakers = set()

    if meta_path.exists():
        with open(meta_path, encoding="utf-8", errors="replace") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    spk_id = parts[1].strip()
                    split = parts[4].strip()
                    if split == "dev":
                        dev_speakers.add(spk_id)
                    elif split == "test":
                        test_speakers.add(spk_id)

    # VoxCeleb2 can have multiple directory layouts:
    #   - dev/wav/{speaker}/{video}/{utt}.wav  (dev set, converted)
    #   - wav/{speaker}/{video}/{utt}.wav      (test set, converted)
    #   - dev/aac/{speaker}/{video}/{utt}.m4a  (dev set, original)
    #   - aac/{speaker}/{video}/{utt}.m4a      (test set, original)
    # Prefer wav/ directories over aac/ if both exist (wav is torchaudio-compatible)
    audio_dirs = []
    dev_wav = root / "dev" / "wav"
    dev_aac = root / "dev" / "aac"
    test_wav = root / "wav"
    test_aac = root / "aac"

    # Dev set: prefer wav over aac
    if dev_wav.exists():
        audio_dirs.append((dev_wav, "dev"))
    elif dev_aac.exists():
        audio_dirs.append((dev_aac, "dev"))

    # Test set: prefer wav over aac
    if test_wav.exists():
        audio_dirs.append((test_wav, "test"))
    elif test_aac.exists():
        audio_dirs.append((test_aac, "test"))

    if not audio_dirs:
        return None

    all_speakers = {}
    all_files = []
    for audio_dir, default_label in audio_dirs:
        for spk_dir in sorted(audio_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            spk_files = []
            for video_dir in spk_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                for f in video_dir.iterdir():
                    if f.suffix.lower() in AUDIO_EXTENSIONS:
                        spk_files.append(f)
            all_speakers[spk_id] = all_speakers.get(spk_id, 0) + len(spk_files)
            all_files.extend(spk_files)

    if not all_files:
        return None

    dev_spk = sum(1 for k in all_speakers if k in dev_speakers)
    test_spk = sum(1 for k in all_speakers if k in test_speakers)
    dev_utt = sum(v for k, v in all_speakers.items() if k in dev_speakers)
    test_utt = sum(v for k, v in all_speakers.items() if k in test_speakers)

    # Determine format from audio_dirs used
    dir_names = [d.name for d, _ in audio_dirs]
    fmt_note = "WAV (converted from AAC)" if "wav" in dir_names else "AAC format"

    return [{
        "dataset": "VoxCeleb2",
        "speakers": len(all_speakers),
        "utterances": len(all_files),
        "files": all_files,
        "note": f"{fmt_note}; dev={dev_spk} spk/{dev_utt} utt, test={test_spk} spk/{test_utt} utt",
    }]


def discover_librispeech(root: Path):
    """
    LibriSpeech structure: LibriSpeech/test-{clean,other}/{speaker}/{chapter}/{spk-chap-NNNN}.flac
    """
    ls_dir = root / "LibriSpeech"
    if not ls_dir.exists():
        return None

    all_speakers = {}
    all_files = []
    split_info = []
    for split_name in ["test-clean", "test-other"]:
        split_dir = ls_dir / split_name
        if not split_dir.exists():
            continue
        split_spk = 0
        split_utt = 0
        for spk_dir in sorted(split_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_files = []
            for chap_dir in spk_dir.iterdir():
                if not chap_dir.is_dir():
                    continue
                for f in chap_dir.iterdir():
                    if f.suffix.lower() == ".flac":
                        spk_files.append(f)
            all_speakers[spk_dir.name] = all_speakers.get(spk_dir.name, 0) + len(spk_files)
            all_files.extend(spk_files)
            split_spk += 1
            split_utt += len(spk_files)
        split_info.append(f"{split_name}={split_spk} spk/{split_utt} utt")

    return [{
        "dataset": "LibriSpeech",
        "speakers": len(all_speakers),
        "utterances": len(all_files),
        "files": all_files,
        "note": ", ".join(split_info),
    }]


def discover_vctk(root: Path):
    """
    VCTK structure varies. Common: wav48_silence_trimmed/{speaker}/{spk_NNN_mic1}.flac
    or wav48/{speaker}/{spk_NNN}.wav
    """
    # Try common VCTK subdirectory names
    candidates = [
        root / "VCTK-Corpus-0.92",
        root / "VCTK-Corpus",
        root,
    ]

    vctk_root = None
    for c in candidates:
        if c.exists() and (c / "wav48_silence_trimmed").exists():
            vctk_root = c / "wav48_silence_trimmed"
            break
        if c.exists() and (c / "wav48").exists():
            vctk_root = c / "wav48"
            break

    if vctk_root is None:
        # Fallback: search for wav directories
        for c in candidates:
            if not c.exists():
                continue
            for d in c.rglob("*"):
                if d.is_dir() and d.name.startswith("wav"):
                    vctk_root = d
                    break
            if vctk_root:
                break

    if vctk_root is None:
        return None

    speakers = {}
    all_files = []
    for spk_dir in sorted(vctk_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_files = [f for f in spk_dir.iterdir()
                     if f.suffix.lower() in AUDIO_EXTENSIONS]
        speakers[spk_dir.name] = len(spk_files)
        all_files.extend(spk_files)

    return [{
        "dataset": "VCTK",
        "speakers": len(speakers),
        "utterances": len(all_files),
        "files": all_files,
        "audio_root": str(vctk_root),
    }]


def discover_voices(root: Path):
    """
    VOiCES devkit structure: VOiCES_devkit/source-16k/{speaker_id}/{file}.wav
    or distant recordings in various room/noise conditions.
    """
    # Find the devkit directory
    candidates = [
        root / "VOiCES_devkit",
        root,
    ]

    devkit = None
    for c in candidates:
        if c.exists() and (c / "source-16k").exists():
            devkit = c
            break

    if devkit is None:
        # Search more broadly
        for c in candidates:
            if not c.exists():
                continue
            for d in c.iterdir():
                if d.is_dir() and "voice" in d.name.lower():
                    devkit = d
                    break
            if devkit:
                break

    if devkit is None:
        return None

    # Collect all WAV files recursively
    speakers = defaultdict(list)
    all_files = []
    for wav_file in devkit.rglob("*.wav"):
        # Try to extract speaker ID from path
        parts = wav_file.parts
        # Common patterns: .../sp{NNNN}/... or speaker ID in filename
        spk_id = None
        for p in parts:
            if p.startswith("sp") or p.startswith("Lab41"):
                spk_id = p
                break
        if spk_id is None:
            # Try extracting from filename like sp{NNNN}_...
            name = wav_file.stem
            for token in name.split("_"):
                if token.startswith("sp"):
                    spk_id = token
                    break
        if spk_id is None:
            spk_id = "unknown"
        speakers[spk_id].append(wav_file)
        all_files.append(wav_file)

    return [{
        "dataset": "VOiCES",
        "speakers": len(speakers),
        "utterances": len(all_files),
        "files": all_files,
        "audio_root": str(devkit),
    }]


def discover_cnceleb(root: Path):
    """
    CN-Celeb structure: CN-Celeb/eval/... or data/{speaker_id}/{utt}.wav
    """
    # Find data directory
    candidates = [
        root / "CN-Celeb_flac",
        root / "CN-Celeb_v2",
        root / "cn-celeb_v2",
        root / "CN-Celeb",
        root,
    ]

    data_dir = None
    for c in candidates:
        if not c.exists():
            continue
        # Look for data or eval subdirectories
        if (c / "data").exists():
            data_dir = c / "data"
            break
        if (c / "eval").exists():
            data_dir = c / "eval"
            break
        # Check if speaker directories exist directly
        sub_dirs = [d for d in c.iterdir() if d.is_dir() and d.name.startswith("id")]
        if sub_dirs:
            data_dir = c
            break

    if data_dir is None:
        # Broad search
        for c in candidates:
            if not c.exists():
                continue
            for d in c.rglob("*"):
                if d.is_dir() and d.name in ("data", "eval"):
                    data_dir = d
                    break
            if data_dir:
                break

    if data_dir is None:
        return None

    speakers = {}
    all_files = []
    for spk_dir in sorted(data_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_files = list(spk_dir.rglob("*.wav")) + list(spk_dir.rglob("*.flac"))
        speakers[spk_dir.name] = len(spk_files)
        all_files.extend(spk_files)

    return [{
        "dataset": "CN-Celeb1",
        "speakers": len(speakers),
        "utterances": len(all_files),
        "files": all_files,
        "audio_root": str(data_dir),
    }]


def discover_musan(root: Path):
    """
    MUSAN structure: musan/{music,speech,noise}/{source}/{file}.wav
    """
    musan_dir = root / "musan"
    if not musan_dir.exists():
        return None

    all_files = []
    cat_info = []
    for category in ["music", "speech", "noise"]:
        cat_dir = musan_dir / category
        if not cat_dir.exists():
            continue
        cat_files = list(cat_dir.rglob("*.wav"))
        all_files.extend(cat_files)
        cat_info.append(f"{category}={len(cat_files)}")

    return [{
        "dataset": "MUSAN",
        "speakers": 0,
        "utterances": len(all_files),
        "files": all_files,
        "note": ", ".join(cat_info),
    }]


def discover_rir(root: Path):
    """
    RIR structure: RIRS_NOISES/{simulated_rirs,real_rirs_isotropic_noises,pointsource_noises}/...
    """
    rir_dir = root / "RIRS_NOISES"
    if not rir_dir.exists():
        return None

    all_files = []
    cat_info = []
    for category in ["simulated_rirs", "real_rirs_isotropic_noises", "pointsource_noises"]:
        cat_dir = rir_dir / category
        if not cat_dir.exists():
            continue
        cat_files = list(cat_dir.rglob("*.wav"))
        all_files.extend(cat_files)
        cat_info.append(f"{category}={len(cat_files)}")

    return [{
        "dataset": "RIR",
        "speakers": 0,
        "utterances": len(all_files),
        "files": all_files,
        "note": ", ".join(cat_info),
    }]


# ──────────────────────────────────────────────────────────────
# Audio property sampling
# ──────────────────────────────────────────────────────────────

def sample_audio_properties(files: list, n: int = SAMPLE_SIZE):
    """
    Sample N files and get audio properties: sample_rate, channels, duration, subtype.
    Returns aggregated statistics.
    """
    if not files:
        return {
            "sample_rate": "N/A",
            "channels": "N/A",
            "bit_depth": "N/A",
            "avg_duration_s": 0.0,
            "est_total_hours": 0.0,
            "format": "N/A",
            "sampled": 0,
        }

    sampled = random.sample(files, min(n, len(files)))
    sample_rates = []
    channels_list = []
    durations = []
    subtypes = []
    formats = set()
    errors = 0

    for fp in sampled:
        formats.add(fp.suffix.lower())
        try:
            info = torchaudio.info(str(fp))
            sr = info.sample_rate
            ch = info.num_channels
            dur = info.num_frames / sr if sr > 0 else 0
            sample_rates.append(sr)
            channels_list.append(ch)
            durations.append(dur)
        except Exception:
            try:
                # Fallback to soundfile
                with sf.SoundFile(str(fp)) as f:
                    sr = f.samplerate
                    ch = f.channels
                    dur = len(f) / sr if sr > 0 else 0
                    st = f.subtype
                sample_rates.append(sr)
                channels_list.append(ch)
                durations.append(dur)
                subtypes.append(st)
            except Exception:
                try:
                    # Fallback to ffprobe (for m4a/aac files)
                    r = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_streams",
                         "-of", "json", str(fp)],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0:
                        streams = json.loads(r.stdout)
                        s = streams["streams"][0]
                        sr = int(s.get("sample_rate", 0))
                        ch = int(s.get("channels", 0))
                        dur = float(s.get("duration", 0))
                        sample_rates.append(sr)
                        channels_list.append(ch)
                        durations.append(dur)
                    else:
                        errors += 1
                except Exception:
                    errors += 1

    if not durations:
        return {
            "sample_rate": "N/A",
            "channels": "N/A",
            "bit_depth": "N/A",
            "avg_duration_s": 0.0,
            "est_total_hours": 0.0,
            "format": ", ".join(formats),
            "sampled": len(sampled),
            "errors": errors,
        }

    avg_dur = sum(durations) / len(durations)
    est_total_hours = (avg_dur * len(files)) / 3600.0

    # Most common sample rate and channels
    sr_mode = max(set(sample_rates), key=sample_rates.count)
    ch_mode = max(set(channels_list), key=channels_list.count)

    # Bit depth from subtype if available
    bit_depth = "16-bit"  # default assumption
    if subtypes:
        st_mode = max(set(subtypes), key=subtypes.count)
        if "16" in st_mode:
            bit_depth = "16-bit"
        elif "24" in st_mode:
            bit_depth = "24-bit"
        elif "32" in st_mode:
            bit_depth = "32-bit"
        elif "FLOAT" in st_mode.upper():
            bit_depth = "32-bit float"

    return {
        "sample_rate": sr_mode,
        "channels": ch_mode,
        "bit_depth": bit_depth,
        "avg_duration_s": round(avg_dur, 2),
        "est_total_hours": round(est_total_hours, 1),
        "format": ", ".join(sorted(formats)),
        "sampled": len(sampled),
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VQI Dataset Inventory")
    print("=" * 70)
    print(f"Datasets root: {DATASETS_ROOT}")
    print(f"Output CSV:    {OUTPUT_CSV}")
    print()

    random.seed(42)  # reproducible sampling

    # Discover all datasets
    discoveries = [
        ("voxCELEB1", discover_voxceleb1),
        ("voxCELEB2", discover_voxceleb2),
        ("librispeech", discover_librispeech),
        ("VCTK", discover_vctk),
        ("VOiCES", discover_voices),
        ("CN-Celeb", discover_cnceleb),
        ("MUSAN", discover_musan),
        ("RIR", discover_rir),
    ]

    all_results = []

    for dir_name, discover_fn in discoveries:
        dataset_path = DATASETS_ROOT / dir_name
        print(f"Scanning {dir_name}...", end=" ", flush=True)

        if not dataset_path.exists():
            print("NOT FOUND")
            all_results.append({
                "dataset": dir_name,
                "status": "NOT FOUND",
                "speakers": 0,
                "utterances": 0,
            })
            continue

        t0 = time.time()
        try:
            entries = discover_fn(dataset_path)
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "dataset": dir_name,
                "status": f"ERROR: {e}",
                "speakers": 0,
                "utterances": 0,
            })
            continue

        if entries is None:
            print("NOT EXTRACTED or EMPTY")
            all_results.append({
                "dataset": dir_name,
                "status": "NOT EXTRACTED",
                "speakers": 0,
                "utterances": 0,
            })
            continue

        for entry in entries:
            files = entry.pop("files", [])
            props = sample_audio_properties(files)
            entry.update(props)
            entry["status"] = "OK"
            all_results.append(entry)

        elapsed = time.time() - t0
        names = [e["dataset"] for e in entries]
        print(f"done ({elapsed:.1f}s) -> {', '.join(names)}")

    # ── Print summary table ──
    print()
    print("=" * 70)
    print("INVENTORY SUMMARY")
    print("=" * 70)
    header = f"{'Dataset':<25} {'Status':<15} {'Speakers':>8} {'Utts':>10} {'SR':>6} {'Ch':>3} {'AvgDur':>7} {'EstHrs':>8} {'Format':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r.get('dataset','?'):<25} "
            f"{r.get('status','?'):<15} "
            f"{r.get('speakers',0):>8} "
            f"{r.get('utterances',0):>10} "
            f"{str(r.get('sample_rate','?')):>6} "
            f"{str(r.get('channels','?')):>3} "
            f"{r.get('avg_duration_s',0):>7.1f} "
            f"{r.get('est_total_hours',0):>8.1f} "
            f"{r.get('format','?'):<10}"
        )
    print()

    # ── Write CSV ──
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "status", "speakers", "utterances",
        "sample_rate", "channels", "bit_depth", "avg_duration_s",
        "est_total_hours", "format", "sampled", "errors", "note", "audio_root",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"CSV written to: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
