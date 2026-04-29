from __future__ import annotations

import argparse
import json
from pathlib import Path

from mscoco_vqa_common import SPLIT_SPECS, default_mscoco_root, image_relative_path, load_json


def count_jpgs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.jpg"))


def verify_split(root: Path, split: str) -> dict:
    spec = SPLIT_SPECS[split]
    question_path = root / spec.question_file
    annotation_path = root / spec.annotation_file
    image_dir = root / spec.image_dir

    questions = load_json(question_path)["questions"]
    annotations = load_json(annotation_path)["annotations"]
    qids = {int(item["question_id"]) for item in questions}
    ann_qids = {int(item["question_id"]) for item in annotations}
    expected_images = {
        image_relative_path(spec, int(item["image_id"]))
        for item in questions
    }
    missing_images = [
        image for image in sorted(expected_images)
        if not (root / image).exists()
    ]

    return {
        "split": split,
        "question_count": len(questions),
        "annotation_count": len(annotations),
        "unique_images_in_questions": len(expected_images),
        "image_files": count_jpgs(image_dir),
        "missing_annotation_qids": len(qids - ann_qids),
        "extra_annotation_qids": len(ann_qids - qids),
        "missing_images": len(missing_images),
        "missing_image_preview": missing_images[:10],
        "ok": (
            len(qids - ann_qids) == 0
            and len(ann_qids - qids) == 0
            and len(missing_images) == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MSCOCO VQA train/val assets.")
    parser.add_argument("--mscoco-root", type=Path, default=default_mscoco_root())
    args = parser.parse_args()

    required = []
    for spec in SPLIT_SPECS.values():
        required.extend([args.mscoco_root / spec.image_dir, args.mscoco_root / spec.question_file, args.mscoco_root / spec.annotation_file])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"ok": False, "missing_required_paths": missing}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    splits = [verify_split(args.mscoco_root, split) for split in ["train", "val"]]
    summary = {
        "mscoco_root": str(args.mscoco_root),
        "splits": splits,
        "ok": all(item["ok"] for item in splits),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
