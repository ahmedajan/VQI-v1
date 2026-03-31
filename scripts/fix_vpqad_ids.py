"""Fix VPQAD Cafeteria subject IDs based on ECAPA2 speaker matching results.

Renames Cafeteria files so that the same physical person has the same subject
ID in both Lab and Cafeteria sessions. Unmatched Cafeteria subjects get new
IDs starting from sub051. Also updates the metadata Excel file.
"""

import os
import shutil
import openpyxl

VPQAD_ROOT = "D:/vqi/Datasets/VPQAD"
CAF_ROOT = os.path.join(VPQAD_ROOT, "Cafeteria_Data")

# Mutual best matches from ECAPA2: Lab_ID -> Caf_ID
# This means Caf_ID should be renamed TO Lab_ID
MATCHES = [
    ("sub001", "sub004"),
    ("sub002", "sub009"),
    ("sub003", "sub005"),
    ("sub004", "sub007"),
    ("sub006", "sub010"),
    ("sub007", "sub016"),
    ("sub008", "sub003"),
    ("sub011", "sub006"),
    ("sub012", "sub008"),
    ("sub014", "sub021"),
    ("sub016", "sub012"),
    ("sub017", "sub024"),
    ("sub018", "sub022"),
    ("sub019", "sub025"),
    ("sub020", "sub011"),
    ("sub021", "sub013"),
    ("sub022", "sub023"),
    ("sub023", "sub001"),
    ("sub024", "sub002"),
    ("sub027", "sub014"),
    ("sub031", "sub017"),
    ("sub032", "sub020"),
    ("sub038", "sub018"),
    ("sub041", "sub027"),
    ("sub042", "sub026"),
    ("sub043", "sub033"),
    ("sub045", "sub029"),
    ("sub047", "sub028"),
    ("sub048", "sub031"),
]

# Unmatched Cafeteria subjects -> new IDs
UNMATCHED_CAF = ["sub015", "sub019", "sub030", "sub032"]
NEW_IDS = ["sub051", "sub052", "sub053", "sub054"]

def build_rename_map():
    """Build caf_old_id -> new_id mapping."""
    rename_map = {}
    for lab_id, caf_id in MATCHES:
        rename_map[caf_id] = lab_id
    for old_id, new_id in zip(UNMATCHED_CAF, NEW_IDS):
        rename_map[old_id] = new_id
    return rename_map


def rename_files(rename_map):
    """Rename Cafeteria audio files using a two-pass approach to avoid collisions."""
    subdirs = ["TD", "TID"]
    total_renamed = 0

    for subdir in subdirs:
        dirpath = os.path.join(CAF_ROOT, subdir)
        if not os.path.isdir(dirpath):
            continue

        # Pass 1: rename to temporary names to avoid collisions
        # e.g., sub001 -> sub023, but sub023 -> sub001 would collide
        temp_renames = []
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".wav"):
                continue
            old_subject = fname.split("_")[0]
            if old_subject in rename_map:
                new_subject = rename_map[old_subject]
                if old_subject == new_subject:
                    continue  # no rename needed
                new_fname = fname.replace(old_subject, f"__TEMP_{new_subject}__", 1)
                old_path = os.path.join(dirpath, fname)
                temp_path = os.path.join(dirpath, new_fname)
                temp_renames.append((temp_path, fname, new_subject))
                os.rename(old_path, temp_path)

        # Pass 2: rename from temp to final
        for temp_path, orig_fname, new_subject in temp_renames:
            old_subject = orig_fname.split("_")[0]
            final_fname = orig_fname.replace(old_subject, new_subject, 1)
            final_path = os.path.join(dirpath, final_fname)
            os.rename(temp_path, final_path)
            total_renamed += 1
            print(f"  {subdir}/{orig_fname} -> {final_fname}")

    return total_renamed


def update_metadata(rename_map):
    """Update the metadata Excel file with corrected subject IDs."""
    meta_path = os.path.join(VPQAD_ROOT, "MetaData", "Metadata.xlsx")
    backup_path = os.path.join(VPQAD_ROOT, "MetaData", "Metadata_ORIGINAL_BACKUP.xlsx")

    # Create backup
    if not os.path.exists(backup_path):
        shutil.copy2(meta_path, backup_path)
        print(f"\nBackup saved: {backup_path}")

    wb = openpyxl.load_workbook(meta_path)
    ws = wb["Cafeteria_Data"]

    updated = 0
    for row in ws.iter_rows(min_row=2):
        old_id = row[0].value
        if old_id in rename_map:
            new_id = rename_map[old_id]
            if old_id != new_id:
                row[0].value = new_id
                updated += 1
                print(f"  Metadata: {old_id} -> {new_id}")

    wb.save(meta_path)
    return updated


def verify_results(rename_map):
    """Verify all renames were applied correctly."""
    subdirs = ["TD", "TID"]
    errors = []

    # Check that no old IDs remain (except those that map to themselves)
    identity_maps = {k for k, v in rename_map.items() if k == v}
    expected_new_ids = set(rename_map.values())

    for subdir in subdirs:
        dirpath = os.path.join(CAF_ROOT, subdir)
        if not os.path.isdir(dirpath):
            continue
        found_ids = set()
        for fname in os.listdir(dirpath):
            if fname.endswith(".wav"):
                sid = fname.split("_")[0]
                found_ids.add(sid)

        # Check no old IDs remain
        old_ids_remaining = set(rename_map.keys()) - expected_new_ids
        stale = found_ids & old_ids_remaining
        if stale:
            errors.append(f"{subdir}: old IDs still present: {stale}")

    return errors


def main():
    print("=" * 70)
    print("VPQAD Subject ID Fix")
    print("=" * 70)

    rename_map = build_rename_map()

    print(f"\nRename mapping ({len(rename_map)} entries):")
    for old_id in sorted(rename_map.keys()):
        new_id = rename_map[old_id]
        label = "(matched to Lab)" if new_id.startswith("sub0") and int(new_id[3:]) <= 50 else "(new unique ID)"
        print(f"  Caf {old_id} -> {new_id} {label}")

    print(f"\n--- Renaming files ---")
    n_renamed = rename_files(rename_map)
    print(f"\nTotal files renamed: {n_renamed}")

    print(f"\n--- Updating metadata ---")
    n_updated = update_metadata(rename_map)
    print(f"Metadata rows updated: {n_updated}")

    print(f"\n--- Verification ---")
    errors = verify_results(rename_map)
    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print(f"  {e}")
    else:
        print("All verifications PASSED")

    # Print final state
    print(f"\n--- Final Cafeteria subject IDs ---")
    for subdir in ["TD", "TID"]:
        dirpath = os.path.join(CAF_ROOT, subdir)
        ids = sorted(set(f.split("_")[0] for f in os.listdir(dirpath) if f.endswith(".wav")))
        print(f"  {subdir}: {len(ids)} subjects -> {ids}")

    # Print final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"29 Cafeteria subjects renamed to match Lab IDs")
    print(f"4 Cafeteria-only subjects assigned new IDs: {UNMATCHED_CAF} -> {NEW_IDS}")
    print(f"Total unique speakers across both sessions: 54")
    print(f"Original metadata backed up to: MetaData/Metadata_ORIGINAL_BACKUP.xlsx")


if __name__ == "__main__":
    main()
