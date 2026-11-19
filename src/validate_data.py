from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    TODO: implement the checks and return a JSON-serializable report.
    The report must not contain timestamps because it is a DVC output.
    """
    report = {"status": "PASS", "total_images": 0, "splits": {}, "errors": []}
    for split, expected_count in EXPECTED_SPLITS.items():
        split_path = raw_root / split
        if not split_path.is_dir():
            report["errors"].append(f"Missing split: {split}")
            continue

        classes = sorted(d for d in split_path.iterdir() if d.is_dir())
        class_names = {directory.name for directory in classes}
        if class_names != EXPECTED_CLASSES:
            report["errors"].append(
                f"Split '{split}' has unexpected classes: {sorted(class_names)}"
            )

        class_counts = {}
        for cls in classes:
            images = sorted(
                path
                for path in cls.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            class_counts[cls.name] = len(images)
            if not images:
                report["errors"].append(
                    f"Class '{cls.name}' in split '{split}' is empty"
                )
            for img_path in images:
                with Image.open(img_path) as img:
                    img.load()

        image_count = sum(class_counts.values())
        report["splits"][split] = {
            "images": image_count,
            "class_counts": class_counts,
        }
        report["total_images"] += image_count
        if image_count != expected_count:
            report["errors"].append(
                f"Split '{split}' has {image_count} images, expected {expected_count}"
            )

    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
