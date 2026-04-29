from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mscoco_vqa_common import load_json, write_json


def load_json_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def find_metadata_path(shard_dir: Path, metadata_name: str) -> Path:
    candidate = shard_dir / metadata_name
    if candidate.exists():
        return candidate
    matches = sorted((shard_dir / "data").glob("mscoco_*.json"))
    matches = [path for path in matches if path.name != "answer_list.json"]
    if not matches:
        return candidate
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently generated MSCOCO VQAttack shard outputs.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--image-subdir", default="attack_outputs/attack_dir")
    parser.add_argument("--text-name", default="attack_outputs/adv_txt.json")
    parser.add_argument("--metadata-name", default="data/mscoco_train_val.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merged_image_dir = args.output_dir / "attack_outputs" / "attack_dir_full"
    merged_image_dir.mkdir(parents=True, exist_ok=True)
    merged_text: dict[str, str] = {}
    merged_metadata: list[dict] = []

    shard_summaries = []
    for shard_index in range(args.num_shards):
        shard_dir = args.shard_root / f"shard_{shard_index:02d}"
        text_path = shard_dir / args.text_name
        metadata_path = find_metadata_path(shard_dir, args.metadata_name)
        image_dir = shard_dir / args.image_subdir

        shard_text = load_json_or_empty(text_path)
        for qid, text in shard_text.items():
            if qid in merged_text:
                raise ValueError(f"Duplicate question_id in shard text outputs: {qid}")
            merged_text[str(qid)] = text

        shard_metadata = []
        if metadata_path.exists():
            shard_metadata = load_json(metadata_path)
            merged_metadata.extend(shard_metadata)

        copied_images = 0
        if image_dir.exists():
            for image_tensor in image_dir.glob("*.pt"):
                target = merged_image_dir / image_tensor.name
                if target.exists() and not args.overwrite:
                    raise FileExistsError(target)
                shutil.copy2(image_tensor, target)
                copied_images += 1

        shard_summaries.append(
            {
                "shard": shard_index,
                "text_count": len(shard_text),
                "metadata_count": len(shard_metadata),
                "tensor_count": copied_images,
                "text_path": str(text_path),
                "metadata_path": str(metadata_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_text_path = args.output_dir / "attack_outputs" / "adv_txt_full.json"
    merged_metadata_path = args.output_dir / "data" / "mscoco_train_val_full.json"
    write_json(merged_text_path, merged_text)
    write_json(merged_metadata_path, merged_metadata)

    summary = {
        "merged_text": str(merged_text_path),
        "merged_image_dir": str(merged_image_dir),
        "merged_metadata": str(merged_metadata_path),
        "text_count": len(merged_text),
        "metadata_count": len(merged_metadata),
        "shards": shard_summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
