from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from mscoco_vqa_common import (
    SPLIT_SPECS,
    default_mscoco_root,
    image_relative_path,
    load_json,
    normalize_splits,
    write_json,
)


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


def select_image_qids(metadata: list[dict], policy: str) -> dict[str, str]:
    qids_by_image: dict[str, list[str]] = defaultdict(list)
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


def update_question_payload(payload: dict, adv_text: dict[str, str], split: str) -> tuple[dict, list[str]]:
    missing: list[str] = []
    for item in payload["questions"]:
        qid = str(int(item["question_id"]))
        if qid not in adv_text:
            missing.append(qid)
            continue
        item["original_question"] = item["question"]
        item["question"] = adv_text[qid]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{split} split is missing attacked question text for {len(missing)} question(s). "
            f"First missing question_id(s): {preview}"
        )
    return payload, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write attacked MSCOCO VQA back to original-format-like files. "
            "Questions are updated, annotations are copied, and image names/counts are preserved."
        )
    )
    parser.add_argument("--mscoco-root", type=Path, default=default_mscoco_root())
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--adv-text-json", type=Path, required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--image-selection", choices=["first", "last"], default="first")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = normalize_splits(args.splits)
    metadata = load_json(args.metadata_json)
    adv_text = {str(int(key)): value for key, value in load_json(args.adv_text_json).items()}
    selected_image_qids = select_image_qids(metadata, args.image_selection)
    metadata_qids = {str(int(record["question_id"])) for record in metadata}
    missing_text_qids = sorted(metadata_qids - set(adv_text), key=lambda value: int(value))
    if missing_text_qids:
        preview = ", ".join(missing_text_qids[:10])
        raise ValueError(
            f"Missing attacked question text for {len(missing_text_qids)} metadata question_id(s): {preview}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(args.output_dir),
        "image_selection": args.image_selection,
        "splits": [],
    }

    for split in splits:
        spec = SPLIT_SPECS[split]
        source_questions = args.mscoco_root / spec.question_file
        source_annotations = args.mscoco_root / spec.annotation_file
        question_payload = load_json(source_questions)
        question_payload, _ = update_question_payload(question_payload, adv_text, split)

        output_questions = args.output_dir / spec.question_file
        output_annotations = args.output_dir / spec.annotation_file
        write_json(output_questions, question_payload)
        output_annotations.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_annotations, output_annotations)

        image_paths = sorted(
            {
                image_relative_path(spec, int(item["image_id"]))
                for item in question_payload["questions"]
            }
        )
        written_images = 0
        for image_path in image_paths:
            selected_qid = selected_image_qids.get(image_path)
            tensor_path = args.tensor_dir / f"{selected_qid}.pt" if selected_qid else None
            if tensor_path is None or not tensor_path.exists():
                raise FileNotFoundError(
                    f"Missing attacked tensor for {image_path} selected_qid={selected_qid}"
                )
            save_image(tensor_path, args.output_dir / image_path, args.overwrite)
            written_images += 1

        summary["splits"].append(
            {
                "split": split,
                "source_questions": str(source_questions),
                "source_annotations": str(source_annotations),
                "output_questions": str(output_questions),
                "output_annotations": str(output_annotations),
                "question_count": len(question_payload["questions"]),
                "unique_images": len(image_paths),
                "attacked_images_written": written_images,
            }
        )

    write_json(args.output_dir / "attack_original_format_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
