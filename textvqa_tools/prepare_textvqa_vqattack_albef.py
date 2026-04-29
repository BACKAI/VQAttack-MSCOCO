import argparse
import json
from collections import Counter
from pathlib import Path


def load_textvqa(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"]


def canonical_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(str(answer).strip().lower() for answer in answers if str(answer).strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def convert_records(records: list[dict]) -> list[dict]:
    converted = []
    for item in records:
        answer = canonical_answer(item.get("answers", []))
        if not answer:
            continue
        converted.append(
            {
                "dataset": "vqa",
                "image": item["image_path"],
                "question": item["question"],
                "question_id": int(item["question_id"]),
                "answer": [str(answer).strip().lower() for answer in item.get("answers", []) if str(answer).strip()],
                "canonical_answer": answer,
                "source_image_id": item["image_id"],
                "source_split": item.get("set_name", ""),
            }
        )
    return converted


def select_shard(records: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return records
    return [record for index, record in enumerate(records) if index % num_shards == shard_index]


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
        f"right_part_textvqa_{tag}.txt": "\n".join(qids) + ("\n" if qids else ""),
        f"albef_ans_table_textvqa_{tag}.json": json.dumps(canonical, ensure_ascii=False, indent=2),
        f"vilt_ans_table_textvqa_{tag}.json": json.dumps(canonical, ensure_ascii=False, indent=2),
        f"all_correct_ans_textvqa_{tag}.json": json.dumps(all_correct, ensure_ascii=False, indent=2),
        f"chatgpt_identity_textvqa_{tag}.json": json.dumps(identity_paraphrase, ensure_ascii=False, indent=2),
    }
    for name, contents in files.items():
        (assets_dir / name).write_text(contents, encoding="utf-8")
    return {key: str(assets_dir / key) for key in files}


def parse_splits(value: str) -> list[str]:
    splits = [split.strip().lower() for split in value.split(",") if split.strip()]
    allowed = {"train", "val", "validation"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"Unsupported split(s): {', '.join(unknown)}")
    return ["val" if split == "validation" else split for split in splits]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TextVQA train/val for ALBEF VQAttack.")
    parser.add_argument(
        "--original-format-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "original_format",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vqattack_albef_smoke100",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated TextVQA splits to include. Supported: train,val. Default: train,val.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output filename tag. Default: smoke<N> when limited, otherwise full.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = parse_splits(args.splits)
    source_records = []
    if "train" in splits:
        source_records.extend(load_textvqa(args.original_format_dir / "TextVQA_0.5.1_train.json"))
    if "val" in splits:
        source_records.extend(load_textvqa(args.original_format_dir / "TextVQA_0.5.1_val.json"))

    limit = None if args.limit <= 0 else args.limit
    default_tag = f"smoke{limit}" if limit is not None else "full"
    if args.num_shards > 1:
        default_tag = f"{default_tag}_shard{args.shard_index:02d}"
    tag = args.tag or default_tag
    records = convert_records(source_records)
    records = select_shard(records, args.num_shards, args.shard_index)
    if limit is not None:
        records = records[:limit]
    answers = {answer for record in records for answer in record["answer"]}

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_tag = "_".join(splits)
    eval_json = data_dir / f"textvqa_{split_tag}_{tag}.json"
    answer_list = data_dir / "answer_list.json"
    eval_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    answer_list.write_text(json.dumps(sorted(answers), ensure_ascii=False, indent=2), encoding="utf-8")

    assets = write_attack_assets(records, args.output_dir, tag)
    manifest = {
        "record_count": len(records),
        "splits": splits,
        "limit": limit,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "tag": tag,
        "image_root": str(args.original_format_dir),
        "eval_json": str(eval_json),
        "answer_list": str(answer_list),
        "assets": assets,
        "notes": [
            "This set uses the requested TextVQA splits in order and excludes test.",
            "Attack assets use ground-truth majority answers as placeholders so VQAttack can run.",
            "For rigorous attack reporting, replace target answer assets with clean ALBEF predictions.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
