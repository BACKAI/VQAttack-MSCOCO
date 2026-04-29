from __future__ import annotations

import argparse
import json
from pathlib import Path

from mscoco_vqa_common import (
    SPLIT_SPECS,
    default_mscoco_root,
    image_relative_path,
    load_json,
    normalize_splits,
)


def count_jpgs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.jpg"))


def validate_split(mscoco_root: Path, output_dir: Path, split: str) -> dict:
    spec = SPLIT_SPECS[split]
    source_questions = load_json(mscoco_root / spec.question_file)
    output_questions = load_json(output_dir / spec.question_file)
    source_annotations = load_json(mscoco_root / spec.annotation_file)
    output_annotations = load_json(output_dir / spec.annotation_file)

    source_question_count = len(source_questions["questions"])
    output_question_count = len(output_questions["questions"])
    source_annotation_count = len(source_annotations["annotations"])
    output_annotation_count = len(output_annotations["annotations"])

    expected_images = {
        image_relative_path(spec, int(item["image_id"]))
        for item in source_questions["questions"]
    }
    missing_original_question = sum(
        1 for item in output_questions["questions"] if "original_question" not in item
    )
    unchanged_questions = sum(
        1
        for item in output_questions["questions"]
        if item.get("original_question") == item.get("question")
    )
    missing_images = [
        image for image in sorted(expected_images) if not (output_dir / image).exists()
    ]

    return {
        "split": split,
        "source_question_count": source_question_count,
        "output_question_count": output_question_count,
        "source_annotation_count": source_annotation_count,
        "output_annotation_count": output_annotation_count,
        "expected_unique_images": len(expected_images),
        "output_image_files": count_jpgs(output_dir / spec.image_dir),
        "missing_original_question": missing_original_question,
        "unchanged_questions": unchanged_questions,
        "missing_image_count": len(missing_images),
        "missing_image_preview": missing_images[:10],
        "ok": (
            source_question_count == output_question_count
            and source_annotation_count == output_annotation_count
            and len(missing_images) == 0
            and missing_original_question == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final attacked MSCOCO VQA output.")
    parser.add_argument("--mscoco-root", type=Path, default=default_mscoco_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--fail-on-unchanged", action="store_true")
    args = parser.parse_args()

    results = [
        validate_split(args.mscoco_root, args.output_dir, split)
        for split in normalize_splits(args.splits)
    ]
    ok = all(item["ok"] for item in results)
    if args.fail_on_unchanged:
        ok = ok and all(item["unchanged_questions"] == 0 for item in results)

    summary = {
        "mscoco_root": str(args.mscoco_root),
        "output_dir": str(args.output_dir),
        "splits": results,
        "ok": ok,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
