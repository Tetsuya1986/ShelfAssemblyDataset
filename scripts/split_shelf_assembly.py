import os
import glob
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split Shelf Assembly Dataset into train/test sets.")
    parser.add_argument("--motion_dir", type=str, help="Directory containing .npz feature files.")
    parser.add_argument("--out_dir", type=str, help="Output directory for split lists.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to reserve for testing (default: 0.1 for 9:1 split).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Scanning for feature files in {args.motion_dir}...")
    file_pattern = os.path.join(args.motion_dir, "*.npz")
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Error: No .npz files found in {args.motion_dir}")
        return

    # Extract unique identifiers (`no`) based on filename prefix (first 6 characters)
    # This directly matches how dataset.py parses no = int(filename[:6])
    unique_ids = set()
    for filepath in files:
        filename = os.path.basename(filepath)
        # Check condition matching dataset.py: filename[17:19] == "HH"
        if len(filename) >= 19 and filename[17:19] == "HH":
            try:
                no = int(filename[:6])
                # We save strings formatted to 6 chars with leading zeros
                unique_ids.add(f"{no:06d}")
            except ValueError:
                continue
    
    unique_ids = sorted(list(unique_ids))
    total_samples = len(unique_ids)
    print(f"Found {total_samples} unique sequence identifiers matching 'HH' pattern.")

    if total_samples == 0:
        print("No valid sequences found. Exiting.")
        return

    # Shuffle for splitting
    random.shuffle(unique_ids)

    # Calculate split point
    test_count = int(total_samples * args.test_ratio)
    train_count = total_samples - test_count

    test_ids = sorted(unique_ids[:test_count])
    train_ids = sorted(unique_ids[test_count:])

    print(f"Splitting into {train_count} (train) and {test_count} (test).")

    os.makedirs(args.out_dir, exist_ok=True)
    
    train_path = os.path.join(args.out_dir, "train.txt")
    with open(train_path, "w") as f:
        f.write("\n".join(train_ids) + "\n")
    print(f"Saved: {train_path}")

    test_path = os.path.join(args.out_dir, "test.txt")
    with open(test_path, "w") as f:
        f.write("\n".join(test_ids) + "\n")
    print(f"Saved: {test_path}")

    print("Done!")

if __name__ == "__main__":
    main()
