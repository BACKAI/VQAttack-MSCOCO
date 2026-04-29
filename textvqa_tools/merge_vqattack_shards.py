import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently generated VQAttack shard outputs.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--image-subdir", default="attack_outputs/attack_dir")
    parser.add_argument("--text-name", default="attack_outputs/adv_txt.json")
    parser.add_argument("--metadata-name", default="data/textvqa_train_val.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merged_image_dir = args.output_dir / "attack_outputs" / "attack_dir_full"
    merged_image_dir.mkdir(parents=True, exist_ok=True)
    merged_text = {}
    merged_metadata = []

    for shard_index in range(args.num_shards):
        shard_dir = args.shard_root / f"shard_{shard_index:02d}"
        text_path = shard_dir / args.text_name
        metadata_path = shard_dir / args.metadata_name
        if not metadata_path.exists():
            matches = sorted((shard_dir / "data").glob("textvqa_train_val_*.json"))
            metadata_path = matches[0] if matches else metadata_path
        image_dir = shard_dir / args.image_subdir

        for qid, text in load_json(text_path).items():
            if qid in merged_text:
                raise ValueError(f"Duplicate question_id in shard text outputs: {qid}")
            merged_text[qid] = text

        if metadata_path.exists():
            merged_metadata.extend(json.loads(metadata_path.read_text(encoding="utf-8")))

        if image_dir.exists():
            for image_tensor in image_dir.glob("*.pt"):
                target = merged_image_dir / image_tensor.name
                if target.exists() and not args.overwrite:
                    raise FileExistsError(target)
                shutil.copy2(image_tensor, target)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_text_path = args.output_dir / "attack_outputs" / "adv_txt_full.json"
    merged_text_path.parent.mkdir(parents=True, exist_ok=True)
    merged_text_path.write_text(json.dumps(merged_text, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_metadata_path = args.output_dir / "data" / "textvqa_train_val_full.json"
    merged_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata_path.write_text(json.dumps(merged_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "merged_text": str(merged_text_path),
                "merged_image_dir": str(merged_image_dir),
                "merged_metadata": str(merged_metadata_path),
                "text_count": len(merged_text),
                "metadata_count": len(merged_metadata),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
