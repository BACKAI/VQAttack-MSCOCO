import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


SPLIT_FILES = {
    "train": "TextVQA_0.5.1_train.json",
    "val": "TextVQA_0.5.1_val.json",
    "validation": "TextVQA_0.5.1_val.json",
}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.size(0) not in {1, 3}:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")

    tensor = tensor.detach().cpu().float()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    tensor = (tensor * 255).round().byte()

    if tensor.size(0) == 1:
        return Image.fromarray(tensor.squeeze(0).numpy(), mode="L").convert("RGB")
    return Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB")


def question_tokens(question: str) -> list[str]:
    return question.replace("?", " ?").replace(".", " .").replace(",", " ,").lower().split()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_splits(raw_splits: str) -> list[str]:
    splits = []
    for split in raw_splits.split(","):
        split = split.strip().lower()
        if not split:
            continue
        if split not in SPLIT_FILES:
            raise ValueError(f"Unsupported split: {split}")
        normalized = "val" if split == "validation" else split
        if normalized not in splits:
            splits.append(normalized)
    return splits


def select_image_qids(metadata: list[dict], policy: str) -> dict[str, str]:
    qids_by_image = defaultdict(list)
    for record in metadata:
        qids_by_image[record["image"]].append(str(record["question_id"]))

    selected = {}
    for image_path, qids in qids_by_image.items():
        if policy == "first":
            selected[image_path] = qids[0]
        elif policy == "last":
            selected[image_path] = qids[-1]
        else:
            raise ValueError(f"Unsupported image selection policy: {policy}")
    return selected


def save_image(tensor_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_image(torch.load(tensor_path, map_location="cpu"))
    save_kwargs = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = 95
    image.save(output_path, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write attacked TextVQA back to original_format-like files. "
            "All questions are updated in JSON. Images keep original names and counts."
        )
    )
    parser.add_argument("--original-format-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--adv-text-json", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--image-selection", choices=["first", "last"], default="first")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = load_json(args.metadata_json)
    adv_text = {str(key): value for key, value in load_json(args.adv_text_json).items()}
    selected_image_qids = select_image_qids(metadata, args.image_selection)
    metadata_qids = {str(record["question_id"]) for record in metadata}
    missing_text_qids = sorted(metadata_qids - set(adv_text), key=lambda value: int(value))
    if missing_text_qids:
        raise ValueError(f"Missing attacked question text for {len(missing_text_qids)} question_id(s)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(args.output_dir),
        "image_selection": args.image_selection,
        "splits": [],
    }

    for split in normalize_splits(args.splits):
        source_json = args.original_format_dir / SPLIT_FILES[split]
        payload = load_json(source_json)
        split_data = payload["data"]
        updated_questions = 0

        for item in split_data:
            qid = str(item["question_id"])
            if qid not in adv_text:
                raise ValueError(f"Missing attacked question text for question_id={qid}")
            item["original_question"] = item["question"]
            item["question"] = adv_text[qid]
            item["question_tokens"] = question_tokens(adv_text[qid])
            updated_questions += 1

        output_json = args.output_dir / SPLIT_FILES[split]
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        image_paths = sorted({item["image_path"] for item in split_data})
        written_images = 0
        for image_path in image_paths:
            output_image = args.output_dir / image_path
            selected_qid = selected_image_qids.get(image_path)
            tensor_path = args.tensor_dir / f"{selected_qid}.pt" if selected_qid is not None else None

            if tensor_path is not None and tensor_path.exists():
                save_image(tensor_path, output_image, args.overwrite)
                written_images += 1
            else:
                raise FileNotFoundError(f"Missing attacked tensor for {image_path} selected_qid={selected_qid}")

        summary["splits"].append(
            {
                "split": split,
                "source_json": str(source_json),
                "output_json": str(output_json),
                "question_count": len(split_data),
                "updated_questions": updated_questions,
                "unique_images": len(image_paths),
                "attacked_images_written": written_images,
            }
        )

    manifest_path = args.output_dir / "attack_original_format_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
