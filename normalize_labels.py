"""
normalize_labels.py — Convert raw pixel-coordinate YOLO labels to normalized format.

The whole_pipeline_annotation_5class folder contains labels with absolute pixel
coordinates (e.g., "4 451 355 547 451") instead of YOLO's expected normalized
format (0-1 range). This script converts them using known image dimensions.

All BBBC041 images are confirmed to be 1600x1200 pixels.

Usage:
    python normalize_labels.py
"""

import os
from pathlib import Path

# Confirmed fixed dimensions for all BBBC041 images
IMG_WIDTH = 1600
IMG_HEIGHT = 1200

# Source: raw pixel-coordinate labels (5-class, but unnormalized)
SOURCE_DIR = Path(
    os.path.expanduser(
        "~/.cache/kagglehub/datasets/khanhtq2101/bbbc041-detection/versions/5/"
        "BBBC041_detection/whole_pipeline_annotation_5class"
    )
)

# Destination: where the project expects normalized labels
DEST_DIR = Path("data/raw/malaria/labels")


def convert_file(src_path: Path, dest_path: Path) -> tuple[int, int]:
    """Convert one label file from pixel coords to normalized YOLO format.

    Source format per line: class_id x_min y_min x_max y_max  (raw pixels)
    Output format per line: class_id x_center y_center width height  (0-1 normalized)

    Returns:
        (lines_written, lines_skipped)
    """
    if not src_path.exists():
        return 0, 0

    lines_out = []
    skipped = 0

    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                skipped += 1
                continue

            class_id = parts[0]
            x_min, y_min, x_max, y_max = map(float, parts[1:5])

            # Clamp to image bounds defensively
            x_min = max(0, min(x_min, IMG_WIDTH))
            x_max = max(0, min(x_max, IMG_WIDTH))
            y_min = max(0, min(y_min, IMG_HEIGHT))
            y_max = max(0, min(y_max, IMG_HEIGHT))

            box_w = x_max - x_min
            box_h = y_max - y_min

            if box_w <= 0 or box_h <= 0:
                skipped += 1
                continue

            x_center = (x_min + box_w / 2) / IMG_WIDTH
            y_center = (y_min + box_h / 2) / IMG_HEIGHT
            norm_w = box_w / IMG_WIDTH
            norm_h = box_h / IMG_HEIGHT

            # Sanity check: all values must be in [0, 1]
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < norm_w <= 1 and 0 < norm_h <= 1):
                skipped += 1
                continue

            lines_out.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    if lines_out:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")

    return len(lines_out), skipped


def main():
    total_written = 0
    total_skipped = 0
    total_files = 0

    for split in ["train", "val", "test"]:
        src_split_dir = SOURCE_DIR / split
        dest_split_dir = DEST_DIR / split

        if not src_split_dir.exists():
            print(f"WARNING: source split not found: {src_split_dir}")
            continue

        txt_files = list(src_split_dir.glob("*.txt"))
        print(f"\nProcessing {split}: {len(txt_files)} label files")

        for src_file in txt_files:
            dest_file = dest_split_dir / src_file.name
            written, skipped = convert_file(src_file, dest_file)
            total_written += written
            total_skipped += skipped
            total_files += 1

    print("\n" + "=" * 50)
    print("NORMALIZATION COMPLETE")
    print("=" * 50)
    print(f"Files processed:     {total_files}")
    print(f"Boxes written:       {total_written}")
    print(f"Boxes skipped:       {total_skipped}")
    print(f"Output directory:    {DEST_DIR}")


if __name__ == "__main__":
    main()