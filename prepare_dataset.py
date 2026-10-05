import os
import shutil
import hashlib
from collections import Counter
from PIL import Image

def prepare_dataset():
    # Source and Target paths
    dataset_b_path = os.path.join("Forest-Fire-Cleaned-dataV2", "Forest-Fire-Cleaned-dataV2")
    target_path = "processed_data"

    if not os.path.exists(dataset_b_path):
        raise FileNotFoundError(f"Source dataset path not found: {dataset_b_path}")

    # Priority mapping for splits (lower number = higher priority)
    split_priority = {'train': 1, 'val': 2, 'test': 3}
    class_map = {'fire': 'Fire', 'nofire': 'Non-Fire'}

    # Track hashes and original counts
    hashes = {}
    orig_counts = Counter()
    
    # 1. Scan and compute SHA-256 hashes across entire Dataset B
    for root, dirs, files in os.walk(dataset_b_path):
        for f in files:
            filepath = os.path.join(root, f)
            relpath = os.path.relpath(filepath, dataset_b_path).replace('\\', '/')
            parts = relpath.split('/')
            
            if len(parts) < 3:
                continue
                
            split = parts[0].lower()
            cls_folder = parts[1].lower()
            cls = 'fire' if cls_folder == 'fire' else ('nofire' if cls_folder in ['nofire', 'no_fire'] else None)
            
            if split not in split_priority or cls not in class_map:
                continue

            orig_counts[(split, cls)] += 1

            # Read file bytes & compute SHA-256
            with open(filepath, 'rb') as fp:
                h = hashlib.sha256(fp.read()).hexdigest()

            if h not in hashes:
                hashes[h] = []
            hashes[h].append({
                'filepath': filepath,
                'relpath': relpath,
                'split': split,
                'class': cls,
                'filename': f
            })

    # 2. Check for conflicting-label duplicate groups (Rule 5)
    conflicting_groups = []
    for h, items in hashes.items():
        classes_in_group = set(item['class'] for item in items)
        if len(classes_in_group) > 1:
            conflicting_groups.append((h, items))

    num_conflicts = len(conflicting_groups)
    if num_conflicts > 0:
        print(f"CRITICAL: Found {num_conflicts} conflicting-label duplicate groups!")
        for h, items in conflicting_groups:
            print(f"Conflict Hash {h[:10]}: {[item['relpath'] for item in items]}")
        print("Stopping dataset preparation as required by Rule 5.")
        return

    # 3. Deduplicate deterministically based on split priority (train > val > test)
    retained_images = []
    removed_per_split = Counter()
    cross_split_duplicates_removed = 0

    for h, items in sorted(hashes.items()):
        # Sort items by split priority, then by relative path for deterministic tie-breaking
        sorted_items = sorted(items, key=lambda x: (split_priority[x['split']], x['relpath']))
        best = sorted_items[0]
        retained_images.append(best)

        best_split = best['split']
        for other in sorted_items[1:]:
            removed_per_split[other['split']] += 1
            if other['split'] != best_split:
                cross_split_duplicates_removed += 1

    # 4. Copy retained images to processed_data structure
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    final_counts = Counter()
    corrupt_count = 0

    for item in retained_images:
        dest_split = item['split']
        dest_class = class_map[item['class']]
        dest_dir = os.path.join(target_path, dest_split, dest_class)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_filepath = os.path.join(dest_dir, item['filename'])
        shutil.copy2(item['filepath'], dest_filepath)
        final_counts[(dest_split, dest_class)] += 1

        # Verify image integrity
        try:
            with Image.open(dest_filepath) as img:
                img.verify()
            with Image.open(dest_filepath) as img:
                img.load()
        except Exception:
            corrupt_count += 1

    # 5. Validation confirmations
    # Check remaining duplicates across processed_data
    processed_hashes = set()
    duplicate_found = False
    for root, dirs, files in os.walk(target_path):
        for f in files:
            fp = os.path.join(root, f)
            with open(fp, 'rb') as file_obj:
                ph = hashlib.sha256(file_obj.read()).hexdigest()
            if ph in processed_hashes:
                duplicate_found = True
            processed_hashes.add(ph)

    all_classes_exist = all(final_counts[(s, c)] > 0 for s in ['train', 'val', 'test'] for c in ['Fire', 'Non-Fire'])

    # 6. Complete Report
    print("=" * 60)
    print("DATASET PREPARATION & DEDUPLICATION REPORT")
    print("=" * 60)
    
    print("\n1. Original Counts (Dataset B):")
    print(f"   train/fire:   {orig_counts[('train', 'fire')]}")
    print(f"   train/nofire: {orig_counts[('train', 'nofire')]}")
    print(f"   val/fire:     {orig_counts[('val', 'fire')]}")
    print(f"   val/nofire:   {orig_counts[('val', 'nofire')]}")
    print(f"   test/fire:    {orig_counts[('test', 'fire')]}")
    print(f"   test/nofire:  {orig_counts[('test', 'nofire')]}")

    print("\n2. Duplicate Files Removed from Each Split:")
    print(f"   train: {removed_per_split['train']}")
    print(f"   val:   {removed_per_split['val']}")
    print(f"   test:  {removed_per_split['test']}")

    print(f"\n3. Number of Cross-Split Duplicates Removed: {cross_split_duplicates_removed}")

    print(f"\n4. Number of Conflicting-Label Duplicate Groups: {num_conflicts}")

    print("\n5. Final processed_data Counts:")
    print(f"   train/Fire:     {final_counts[('train', 'Fire')]}")
    print(f"   train/Non-Fire: {final_counts[('train', 'Non-Fire')]}")
    print(f"   val/Fire:       {final_counts[('val', 'Fire')]}")
    print(f"   val/Non-Fire:   {final_counts[('val', 'Non-Fire')]}")
    print(f"   test/Fire:      {final_counts[('test', 'Fire')]}")
    print(f"   test/Non-Fire:  {final_counts[('test', 'Non-Fire')]}")

    total_final_count = sum(final_counts.values())
    print(f"\n6. Total Final Image Count: {total_final_count}")

    print("\n7. Validations & Confirmations:")
    print(f"   • Zero corrupt/unreadable images: {'CONFIRMED (0 corrupt)' if corrupt_count == 0 else 'FAILED'}")
    print(f"   • Zero exact duplicate hashes across final splits: {'CONFIRMED (0 duplicates)' if not duplicate_found else 'FAILED'}")
    print(f"   • Both classes exist in every split: {'CONFIRMED' if all_classes_exist else 'FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_dataset()
