from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mscoco_vqa_common import (
    SPLIT_SPECS,
    answers_from_annotation,
    canonical_answer,
    default_mscoco_root,
    image_relative_path,
    load_json,
    normalize_splits,
    split_tag,
    write_json,
)


def select_shard(records: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return records
    return [record for index, record in enumerate(records) if index % num_shards == shard_index]


def parse_max_images_per_split(raw_value: str) -> dict[str, int]:
    limits: dict[str, int] = {}
    raw_value = raw_value.strip()
    if not raw_value:
        return limits

    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--max-images-per-split entries must look like split=count, for example val=3166"
            )
        raw_split, raw_count = item.split("=", 1)
        split = normalize_splits(raw_split)[0]
        count = int(raw_count)
        if count < 1:
            raise ValueError("--max-images-per-split counts must be positive")
        limits[split] = count
    return limits


def load_split_records(
    mscoco_root: Path,
    split: str,
    skip_missing_images: bool,
    max_images: int = 0,
) -> list[dict]:
    spec = SPLIT_SPECS[split]
    questions_payload = load_json(mscoco_root / spec.question_file)
    annotations_payload = load_json(mscoco_root / spec.annotation_file)
    annotations_by_qid = {
        int(item["question_id"]): item for item in annotations_payload["annotations"]
    }

    records: list[dict] = []
    missing_annotations: list[int] = []
    missing_images: list[str] = []
    image_exists_cache: dict[str, bool] = {}
    selected_images: set[str] = set()

    for item in questions_payload["questions"]:
        qid = int(item["question_id"])
        annotation = annotations_by_qid.get(qid)
        if annotation is None:
            missing_annotations.append(qid)
            continue

        answers = answers_from_annotation(annotation)
        answer = canonical_answer(annotation, answers)
        if not answers or not answer:
            continue

        image = image_relative_path(spec, int(item["image_id"]))
        if image not in image_exists_cache:
            image_exists_cache[image] = (mscoco_root / image).exists()
        if not image_exists_cache[image]:
            if image not in missing_images:
                missing_images.append(image)
            if skip_missing_images:
                continue

        if max_images > 0 and image not in selected_images:
            if len(selected_images) >= max_images:
                continue
            selected_images.add(image)

        records.append(
            {
                "dataset": "vqa",
                "image": image,
                "question": item["question"],
                "question_id": qid,
                "answer": answers,
                "canonical_answer": answer,
                "source_image_id": int(item["image_id"]),
                "source_split": split,
                "source_question_type": annotation.get("question_type", ""),
                "source_answer_type": annotation.get("answer_type", ""),
            }
        )

    if missing_annotations:
        preview = ", ".join(str(qid) for qid in missing_annotations[:10])
        raise ValueError(
            f"{split} split has {len(missing_annotations)} question(s) without annotation. "
            f"First missing question_id(s): {preview}"
        )
    if missing_images and not skip_missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(
            f"{split} split has {len(missing_images)} missing image file(s). "
            f"First missing image(s): {preview}"
        )

    return records


def write_attack_assets(records: list[dict], output_dir: Path, tag: str) -> dict:
    assets_dir = output_dir / "attack_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    qids = [str(record["question_id"]) for record in records]
    canonical = {str(record["question_id"]): record["canonical_answer"] for record in records}
    all_correct = {str(record["question_id"]): sorted(set(record["answer"])) for record in records}
    identity_paraphrase = {
        str(record["question_id"]): [record["canonical_answer"], record["question"]]
        for record in records
    }

    files = {
        f"right_part_mscoco_{tag}.txt": "\n".join(qids) + ("\n" if qids else ""),
        f"albef_ans_table_mscoco_{tag}.json": json.dumps(canonical, ensure_ascii=False, indent=2),
        f"vilt_ans_table_mscoco_{tag}.json": json.dumps(canonical, ensure_ascii=False, indent=2),
        f"all_correct_ans_mscoco_{tag}.json": json.dumps(all_correct, ensure_ascii=False, indent=2),
        f"chatgpt_identity_mscoco_{tag}.json": json.dumps(identity_paraphrase, ensure_ascii=False, indent=2),
    }
    for name, contents in files.items():
        (assets_dir / name).write_text(contents, encoding="utf-8")
    return {name: str(assets_dir / name) for name in files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MSCOCO VQA v2 train/val for ALBEF VQAttack.")
    parser.add_argument("--mscoco-root", type=Path, default=default_mscoco_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Per-shard record limit. 0 means full shard.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-images-per-split",
        default="",
        help=(
            "Optional comma-separated split image limits, for example val=3166. "
            "The selected images keep all questions."
        ),
    )
    parser.add_argument("--skip-missing-images", action="store_true")
    args = parser.parse_args()

    splits = normalize_splits(args.splits)
    max_images_per_split = parse_max_images_per_split(args.max_images_per_split)
    records: list[dict] = []
    for split in splits:
        records.extend(
            load_split_records(
                args.mscoco_root,
                split,
                args.skip_missing_images,
                max_images=max_images_per_split.get(split, 0),
            )
        )

    records = select_shard(records, args.num_shards, args.shard_index)
    limit = None if args.limit <= 0 else args.limit
    if limit is not None:
        records = records[:limit]

    default_tag = f"smoke{limit}" if limit is not None else "full"
    if args.num_shards > 1:
        default_tag = f"{default_tag}_shard{args.shard_index:02d}"
    tag = args.tag or default_tag

    answer_list = sorted({answer for record in records for answer in record["answer"]})
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_json = data_dir / f"mscoco_{split_tag(splits)}_{tag}.json"
    answer_list_json = data_dir / "answer_list.json"
    write_json(eval_json, records)
    write_json(answer_list_json, answer_list)
    assets = write_attack_assets(records, args.output_dir, tag)

    split_counts = Counter(record["source_split"] for record in records)
    image_counts = Counter()
    for split in splits:
        image_counts[split] = len(
            {record["image"] for record in records if record["source_split"] == split}
        )

    manifest = {
        "record_count": len(records),
        "answer_count": len(answer_list),
        "split_counts": dict(split_counts),
        "image_counts": dict(image_counts),
        "splits": splits,
        "limit": limit,
        "max_images_per_split": max_images_per_split,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "tag": tag,
        "image_root": str(args.mscoco_root),
        "eval_json": str(eval_json),
        "answer_list": str(answer_list_json),
        "assets": assets,
        "notes": [
            "MSCOCO VQA question and annotation files are joined by question_id.",
            "Attack assets use VQA ground-truth multiple_choice_answer as the aligned ALBEF/VILT answer placeholder.",
            "For strict attack reporting, replace target answer assets with clean model predictions before running.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
